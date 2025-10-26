import sounddevice as sd
import numpy as np
import asyncio

async def play_tone(frequency, duration):
    fs = 44100  # Sample rate
    t = np.linspace(0, duration, int(fs * duration), False)  # Time array
    tone = 0.5 * np.sin(frequency * t * 2 * np.pi)  # Generate tone
    await loop.run_in_executor(None, sd.play, tone, fs)  # Play tone

async def main():
    frequencies = [440, 880, 1320]  # Frequencies to play
    duration = 1  # Duration of each tone

    for freq in frequencies:
        await play_tone(freq, duration)
        await asyncio.sleep(duration + 0.5)  # Wait before playing next tone

loop = asyncio.get_event_loop()
loop.run_until_complete(main())
