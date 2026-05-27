# Troubleshooting

## NVIDIA driver / CUDA mismatch with PyTorch

If you see errors about the NVIDIA driver being too old for the installed PyTorch build, align **driver** and **torch** builds using the official [PyTorch install matrix](https://pytorch.org/get-started/locally/).

Example pin used in this project’s README:

```bash
pip uninstall torch
pip install torch==2.9.1
```

Adjust the version to match your CUDA / driver stack.

## SpaceMouse (Linux udev)

Add a rule (example for SpaceMouse Compact) to `/etc/udev/rules.d/99-spacemouse.rules`:

```text
KERNEL=="hidraw*", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666", GROUP="plugdev"
SUBSYSTEM=="usb", ATTRS{idVendor}=="256f", ATTRS{idProduct}=="c635", MODE="0666", GROUP="plugdev"
```

Reload rules:

```bash
sudo udevadm control --reload-rules
```

**pyspacemouse:** `pyspacemouse==1.1.4` is known to work; `2.0.0` has been reported incompatible.

## MediaPipe

Pin **mediapipe==0.10.21** if you hit errors related to `solutions` on newer releases.

## Still stuck?

Open an issue on the [OopsieVerse repository](https://github.com/UT-Austin-RobIn/oopsieverse/issues) with your OS, GPU, driver version, and the exact command plus traceback.
