import asyncio
import logging
import os
import string
from typing import Callable, Set

from livekit.agents import AgentSession

logger = logging.getLogger("interruption-handler")

# Default list of backchannel words (soft acknowledgements)
DEFAULT_IGNORE_WORDS: Set[str] = {
    "yeah", "yes", "yea", "ya", "yep", "yup", "yah", "yeh",
    "ok", "okay", "k", "kk", "alright", "right", "sure", "fine",
    "got it", "i see", "i understand", "understood",
    "hmm", "hm", "hmmmm", "aha", "ah", "uh", "um", "umm",
    "uh-huh", "uh huh", "mhm", "mhmm", "mm", "mmm", "mm-hmm",
    "go on", "continue", "and", "so", "then",
}


def get_ignore_words() -> Set[str]:
    """Load ignore words from environment variable or use defaults."""
    env_words = os.getenv("IGNORE_WORDS")
    if env_words:
        return {w.strip().lower() for w in env_words.split(",")}
    return DEFAULT_IGNORE_WORDS


def sanitize_transcript(text: str) -> str:
    """Normalize transcript text for comparison."""
    text = text.lower().strip()
    # Remove punctuation
    return text.translate(str.maketrans("", "", string.punctuation))


def is_backchannel(transcript: str, ignore_words: Set[str]) -> bool:
    """
    Determine if the transcript is a backchannel (soft acknowledgement).
    
    Returns True if:
    - The transcript is empty
    - The transcript exactly matches an ignore word
    - The transcript is a single word that's in the ignore list
    
    Returns False if:
    - The transcript contains multiple words (potential real interruption)
    - The transcript contains words not in the ignore list
    """
    if not transcript:
        return True

    # Check exact match first
    if transcript in ignore_words:
        return True

    # For multi-word inputs, check if ALL words are in ignore list
    # This handles cases like "yeah ok" vs "yeah but wait"
    words = transcript.split()
    
    # Single word check
    if len(words) == 1:
        return words[0] in ignore_words
    
    # Multi-word: if ANY word is not in ignore list, it's an interruption
    # This handles "yeah wait a second" -> interruption because "wait", "a", "second" are commands
    for word in words:
        if word not in ignore_words and word not in {"a", "the", "but", "and", "or"}:
            return False
    
    return True


class InterruptionHandler:
    """
    Handles intelligent interruption logic for LiveKit agents.
    
    This handler distinguishes between:
    - Backchannel feedback (yeah, ok, hmm) when agent is speaking -> IGNORE
    - Real interruptions (stop, wait, no) when agent is speaking -> INTERRUPT
    - Any input when agent is silent -> PROCESS
    """
    
    def __init__(
        self,
        session: AgentSession,
        ignore_words: Set[str] | None = None,
        on_backchannel: Callable[[str], None] | None = None,
        on_interrupt: Callable[[str], None] | None = None,
    ):
        self.session = session
        self.ignore_words = ignore_words or get_ignore_words()
        self.on_backchannel = on_backchannel
        self.on_interrupt = on_interrupt
        self._registered = False
        
        # Track agent speaking state manually
        self._agent_is_speaking = False
        self._last_agent_speech_time = 0
        
        logger.info(f"[INTERRUPT-HANDLER] Initialized with {len(self.ignore_words)} ignore words")

    def _is_agent_speaking(self) -> bool:
        """
        Check if the agent is currently speaking.
        
        Uses the session's assistant object to check if it's currently speaking.
        Falls back to manual tracking if the attribute is not available.
        """
        # Try to use session's built-in state first
        try:
            if hasattr(self.session, 'assistant') and self.session.assistant:
                # Check if assistant is speaking
                if hasattr(self.session.assistant, 'speaking'):
                    return self.session.assistant.speaking
                # Check if there's active audio output
                if hasattr(self.session.assistant, 'is_speaking'):
                    return self.session.assistant.is_speaking()
        except Exception as e:
            logger.debug(f"Error checking session state: {e}")
        
        # Fallback to manual tracking
        return self._agent_is_speaking

    def _update_agent_state(self, event) -> None:
        """Update the agent speaking state based on events."""
        import time
        
        event_type = event.__class__.__name__
        
        if 'playback' in event_type.lower() or 'speech' in event_type.lower() or 'output' in event_type.lower():
            if 'start' in event_type.lower():
                self._agent_is_speaking = True
                self._last_agent_speech_time = time.time()
                logger.debug(f"[INTERRUPT-HANDLER] Agent started speaking (event: {event_type})")
            elif 'finish' in event_type.lower() or 'end' in event_type.lower():
                self._agent_is_speaking = False
                logger.debug(f"[INTERRUPT-HANDLER] Agent stopped speaking (event: {event_type})")

    def _handle_transcription(self, event) -> None:
        """
        Handle user speech transcription events.
        
        Logic:
        1. Only process final transcriptions (not interim)
        2. If agent is NOT speaking, let normal flow handle it
        3. If agent IS speaking:
           - Check if it's a backchannel -> IGNORE (by suppressing the event)
           - Otherwise -> INTERRUPT (let it through)
        """
        # Only process final transcriptions
        if hasattr(event, 'is_final') and not event.is_final:
            return

        # Get transcript text
        transcript_text = ""
        if hasattr(event, 'transcript'):
            transcript_text = event.transcript
        elif hasattr(event, 'text'):
            transcript_text = event.text
        
        transcript = sanitize_transcript(transcript_text)
        
        # If agent is not speaking, this is normal user input
        # Let the session handle it normally (don't interfere)
        if not self._is_agent_speaking():
            logger.debug(f"[INTERRUPT-HANDLER] Agent silent, passing through: '{transcript_text}'")
            return

        # Agent IS speaking - now we need to decide: backchannel or interrupt?
        
        if not transcript:
            logger.debug("[INTERRUPT-HANDLER] Empty transcript while speaking, ignoring")
            return

        # Check if this is a backchannel (soft acknowledgement)
        if is_backchannel(transcript, self.ignore_words):
            logger.info(f"[INTERRUPT-HANDLER] ✓ Ignored backchannel while speaking: '{transcript_text}'")
            if self.on_backchannel:
                self.on_backchannel(transcript)
            
            # CRITICAL: Mark event as handled to prevent interruption
            # Different LiveKit versions use different approaches
            if hasattr(event, 'cancel'):
                event.cancel()
            if hasattr(event, 'handled'):
                event.handled = True
            if hasattr(event, 'preventDefault'):
                event.preventDefault()
            
            return

        # This is a real interruption
        logger.info(f"[INTERRUPT-HANDLER] ✗ Real interruption detected: '{transcript_text}'")
        if self.on_interrupt:
            self.on_interrupt(transcript)
        
        # Let the normal interruption flow continue

    def register(self) -> "InterruptionHandler":
        """Register event handlers with the session."""
        if not self._registered:
            # Listen to transcription events
            self.session.on("user_input_transcribed", self._handle_transcription)
            
            # Try to register for various state tracking events
            # Different LiveKit versions may have different event names
            for event_name in [
                "agent_speech_started", "agent_speech_committed", "agent_speech_interrupted",
                "agent_started_speaking", "agent_stopped_speaking",
                "playout_started", "playout_stopped",
                "agent_output_started", "agent_output_finished",
            ]:
                try:
                    self.session.on(event_name, self._update_agent_state)
                except Exception:
                    pass  # Event not available in this version
            
            self._registered = True
            logger.info("[INTERRUPT-HANDLER] ✓ Registered event handlers")
        return self

    def unregister(self) -> "InterruptionHandler":
        """Unregister event handlers from the session."""
        if self._registered:
            try:
                self.session.off("user_input_transcribed", self._handle_transcription)
                
                # Unregister state tracking events
                for event_name in [
                    "agent_speech_started", "agent_speech_committed", "agent_speech_interrupted",
                    "agent_started_speaking", "agent_stopped_speaking",
                    "playout_started", "playout_stopped",
                    "agent_output_started", "agent_output_finished",
                ]:
                    try:
                        self.session.off(event_name, self._update_agent_state)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error unregistering handlers: {e}")
            
            self._registered = False
            logger.info("[INTERRUPT-HANDLER] Unregistered event handlers")
        return self


def setup_interruption_handler(
    session: AgentSession,
    ignore_words: Set[str] | None = None,
) -> InterruptionHandler:
    """
    Set up the interruption handler for a LiveKit agent session.
    
    Args:
        session: The AgentSession to attach the handler to
        ignore_words: Optional custom set of words to treat as backchannels
        
    Returns:
        The configured and registered InterruptionHandler instance
    """
    handler = InterruptionHandler(session, ignore_words=ignore_words)
    handler.register()
    return handler