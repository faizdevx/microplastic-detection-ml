# Acquisition protocol for fluorescence images

Images that go into `mp analyse`, the demo app or any new evaluation must be taken in a way
that keeps acquisition from deciding the class. The numbers below come from the generated
results; the rules follow from them.

## Why

- Brightness can be read from the particle features alone: exposure tertile at
  0.850 accuracy (chance 0.334), and still
  0.754 inside a single class (`results/phase5_probe.json`).
- Resolution is readable too: 0.854 against a chance level of
  0.425.
- A classifier trained on discs of class-independent size placed on background pixels reaches
  0.811 balanced accuracy (chance 0.5); with the image-level background statistics it was
  0.906. Shape features on the same discs sit at 0.516, as they should.
- In the archive, PVC and Nylon images differ in size, brightness and background, so the
  recorded accuracy mixes polymer and acquisition (`MODEL_CARD.md`).

## Rules

1. **Frame:** 4:3, with a resolution that appears in the training data (640x480 or 1280x960 are
   the only ones present for both polymers). Other frames are refused by the quality gates and
   get no polymer label.
2. **Fixed settings:** one exposure time, one analogue gain, white balance off, auto-exposure
   off, for every sample of every polymer and every session. Capture with
   `mp.deploy.capture.capture(...)`, which writes the settings next to the image
   (`<image>.json`).
3. **Same optics and illumination:** same objective, filter, UV LED current and distance, same
   slide or cuvette type and the same position in the frame.
4. **Interleave polymers:** photograph PVC and Nylon alternately within one session, so a
   session effect cannot align with a class. Record the session and the order.
5. **Blanks:** take an image of the stained medium without particles at the start and end of
   every session, with the same settings.
6. **Record per image:** sample id, polymer, date and time, session, stain batch, exposure,
   gain, temperature, operator. Keep the raw file; do not resize, crop or re-encode before
   analysis.
7. **Calibration:** photograph a stage micrometer with the same optics once per session and
   run `mp calibrate`; until `um_per_px` is set, sizes stay in pixels.
8. **Check before use:** run `python -m mp.cli analyse --image <file>`; a refusal reason means
   the image does not meet these rules.
