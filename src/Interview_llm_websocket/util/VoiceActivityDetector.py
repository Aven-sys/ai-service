import webrtcvad

class VoiceActivityDetector:
    def __init__(self, mode=3, sample_rate=16000, frame_duration_ms=20):
        """
        Initialize the Voice Activity Detector.
        
        :param mode: Aggressiveness mode (0-3). Higher values are more aggressive.
        :param sample_rate: Audio sample rate, e.g., 16000 Hz.
        :param frame_duration_ms: Frame duration in milliseconds (10, 20, or 30 ms).
        """
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(mode)
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * frame_duration_ms / 1000 * 2)  # 16-bit PCM (2 bytes per sample)

    def is_speech(self, audio_chunk):
        """
        Check if the given audio chunk contains speech.
        
        :param audio_chunk: Audio chunk (bytes).
        :return: True if speech is detected, False otherwise.
        """
        if len(audio_chunk) < self.frame_size:
            raise ValueError(f"Audio chunk size is too small. Expected {self.frame_size} bytes.")
        return self.vad.is_speech(audio_chunk, self.sample_rate)
