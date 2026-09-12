# AuK task guide

[Model downloads and installation](../README.md#models-and-locations)

## Wiring and settings

Select an AuK diffusion model, Qwen encoder and AuK VAE in their loaders. Only model weights need downloading; Qwen tokenizer/config files are bundled in the node pack.

1. Connect Model Loader to both Instruction Encode.model and Generate / Edit.model.
2. Connect Encoder Loader to Instruction Encode.encoder, and VAE Loader to Generate / Edit.vae.
3. Connect Instruction Encode.CONDITIONING to Generate / Edit.conditioning.
4. For direct editing/cloning, connect core Load Audio to Instruction Encode.audio. When using Prompt Enhance, connect Load Audio to Prompt Enhance.audio and connect its prepared_audio output to Instruction Encode.audio. Leave audio disconnected for text-only TTS.
5. Paste an example below into Instruction Encode.instruction, or select its task in Instruction Builder and connect the STRING output. A connected STRING overrides the typed instruction widget.
6. Connect Generate / Edit.AUDIO to core Preview Audio or Save Audio, then queue.

For the local Prompt Enhance workflow, also connect the same Load Audio output to AuK Whisper Transcribe and connect its STRING transcript to Prompt Enhance.context. Qwen2.5-Omni-3B produces the task command locally; the transcript and audio let deterministic PE code calculate content-aware duration and apply the upstream whisper RMS targets.

For difficult editing tasks, our testing found that manually setting the output about **0.2–0.5 seconds shorter than the source** can improve instruction adherence. For example, try 5.5 seconds for a 6-second source, even when the replacement would normally need more time. Disconnect Prompt Enhance.seconds and enter the duration manually to test this workaround.

Start with Base at **32 steps, guidance=2, sway=-1, seed=42**. Flash always uses four fixed steps without guidance and ignores those three sampling controls. `seconds=0` matches the source duration; text-only TTS needs a positive duration. Output is mono 24 kHz. There is no combined source/target duration cap; longer clips require more memory and processing time. Note that upstream's ComfyUI wrapper enforces a shared 30-second source+target budget, while upstream's standalone Python tooling sets no limit — this integration follows the Python tooling.

The loader's `precision` controls compute dtype; the file determines BF16/INT8/W4A8 storage. `attention=auto` follows ComfyUI. Explicit Flash/Sage requests fall back to SDPA for unsupported kernels, devices/dtypes or masks. Use `sdpa` for a comparison baseline.

## Tasks

- [Text-to-speech](#text-to-speech)
- [Voice cloning](#voice-cloning)
- [Speech content editing](#speech-content-editing)
- [Lyric editing](#lyric-editing)
- [Whisper conversion](#whisper-conversion)
- [Emotion and timbre](#emotion-and-timbre)
- [Pitch](#pitch)
- [Speed](#speed)
- [Volume](#volume)
- [Accent removal](#accent-removal)
- [Nonverbal sounds](#nonverbal-sounds)
- [Speech enhancement](#speech-enhancement)
- [Speaker separation](#speaker-separation)
- [Music separation](#music-separation)

## Text-to-speech

[Example workflow](../example_workflows/01_text_to_speech.json)

Leave audio disconnected. Set seconds=3 for the example sentence; increase it if the final words are cut off.

**Voice description TTS**

```text
Generate speech based on the following description: "An adult woman speaking warmly and clearly at a natural pace". The content to speak is: "Welcome home. How was your day?".
```

## Voice cloning

[Example workflow](../example_workflows/03_voice_clone_plus.json)

Connect a clean recording of one speaker to Instruction Encode.audio. Set seconds=5 for the example text; target duration can differ from reference duration.

**Voice cloning**

```text
Say the following with the same voice: "The train leaves at nine. Please meet me at the station.".
```

## Speech content editing

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Connect the speech to edit. The quoted words and anchors must actually occur in that recording. Start with seconds=0 for a similar-length replacement. Allow more time for inserted words and less time for removals.

**Replace speech**

```text
Replace 'Tuesday' with 'Friday'.
```

**Insert speech before**

```text
Add 'Good morning.' before 'Welcome everyone'.
```

**Insert speech after**

```text
Add 'Thank you for coming.' after 'Welcome everyone'.
```

**Remove speech**

```text
Remove 'you know'.
```

## Lyric editing

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Input must be acapella — clean solo singing with no instrumental backing. If the track has accompaniment, extract singing first, then connect that output to the second Instruction Encode.audio for the lyric edit. Change only one or two words per edit, and pick replacement words that sound similar to the originals — similar syllable count and vowel sounds. If the new words do not resemble the old ones, the model will not follow the instruction. Set seconds=0 to retain the melody timing. The workflow does not automatically remix the edited vocal with the accompaniment.

**Edit lyrics**

```text
Change "the sun is rising" to "the moon is shining" in the vocal recording.
```

## Whisper conversion

[Example workflow](../example_workflows/03_voice_clone_plus.json)

Connect the source speech to Instruction Encode.audio and set seconds=0. To diagnose a weak edit, compare Base BF16 + BF16 encoder with attention=sdpa, steps=32, guidance=2 and sway=-1. Check the effective instruction: a connected STRING overrides the typed text. Listen for whispered delivery; reduced volume alone does not establish a successful whisper conversion.

**Convert to whisper**

```text
用小声耳语的方式把这段话说出来。
```

**Whisper to speech**

```text
Convert this whispered speech into a normal speaking voice while preserving the speaker and content.
```

Upstream English alternative for whisper conversion: `Convert this speech into a soft whisper while preserving the speaker and content.`

## Emotion and timbre

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Connect source speech and start with seconds=0. Describe one clear change. Check that the spoken words remain intact; timbre editing deliberately changes the voice quality.

**Change emotion**

```text
Change the emotion to quietly disappointed, with a subdued delivery.
```

**Change timbre**

```text
Keep the spoken content unchanged and change the timbre to: "a mature woman with a low, slightly raspy voice".
```

## Pitch

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Connect source audio and set seconds=0. Start with small pitch changes. These learned edits are not guaranteed to match an exact DSP pitch shift.

**Raise pitch**

```text
Raise the pitch by 2.0 semitones.
```

**Lower pitch**

```text
Lower the pitch by 2.0 semitones.
```

## Speed

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Set seconds to source duration divided by the speed factor. A 10-second source at 1.25x needs seconds=8; at 0.8x it needs seconds=12.5. Keeping seconds=0 can work against the speed instruction.

**Change speed**

```text
Adjust the speech speed to 1.25x.
```

## Volume

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Set seconds=0 and connect Generate / Edit directly to Preview Audio or Save Audio. Avoid normalization afterwards, which can undo the gain change. Check clipping; use a native gain operation when exact dB adjustment is required.

**Increase volume**

```text
Increase the volume by 5.0 dB.
```

**Decrease volume**

```text
Decrease the volume by 5.0 dB.
```

## Accent removal

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Connect the accented speech and set seconds=0. This requests accent removal, not translation. Compare pronunciation and word preservation.

**Remove accent**

```text
Remove the regional accent while preserving the speaker's voice and content.
```

## Nonverbal sounds

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Connect audio containing the sound to remove. For added sounds, allow extra output time. For precise placement, an alternative instruction is: Add a cough before the word hello.

**Remove nonverbal sounds**

```text
Remove all humming from the audio.
```

**Add nonverbal sound**

```text
Add a cough at the beginning of the speech.
```

## Speech enhancement

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Connect the damaged/noisy recording and set seconds=0. Choose the specific defect to fix. Compare quiet words and consonants with the source to check for over-removal.

**Enhance speech**

```text
Preserve all speakers, remove noise and reverberation, and output clean speech of the same length.
```

**Denoise only**

```text
Remove only the background noise, preserve everything else, and output audio of the same length.
```

**Dereverberate only**

```text
Remove only the room reverberation, preserve everything else, and output audio of the same length.
```

**Repair quality**

```text
Repair the telephone effect and restore natural, clear speech.
```

## Speaker separation

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Set seconds=0 to preserve alignment. Speaker order means the order distinct people start talking. For extraction by words, quote a distinctive phrase actually spoken by the desired person.

**Separate speaker**

```text
Keep only the second speaker to start talking and remove all other speakers.
```

**Extract target speaker**

```text
Keep only the speaker who says "Please close the window" and remove all other speakers.
```

## Music separation

[Example workflow](../example_workflows/02_audio_edit_or_clone.json)

Use the first generation stage and preview/save its output directly when no lyric edit is needed. Set seconds=0. Extract singing retains sung vocals; Keep human voices also retains speech. There is no separate instrumental-output socket.

**Extract singing**

```text
Keep only the singing voice and remove everything else.
```

**Keep human voices**

```text
Keep all human voices, including speech and singing, and remove everything else.
```

These are generative edits. Successful execution does not guarantee perfect instruction adherence; compare the result with your source. Task definitions and bilingual templates come from the [upstream cookbook](https://github.com/Tencent-Hunyuan/AuK/blob/main/docs/COOKBOOK.md).
