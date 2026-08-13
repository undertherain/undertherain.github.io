feh, nomacs, Eye of GNOME, Gwenview, qimgv... I've tried dozens of Linux image viewers over the last decade, and something was always missing.

One wouldn't start fullscreen. Another wouldn't close on a single press of Escape. Others felt sluggish, especially when browsing large images from my NAS over Wi-Fi.

Eventually, the pressure built up and I did the obvious unreasonable thing: I built my own.

Meet Better Image View (`biv`): a deliberately small, keyboard-first viewer for Linux.

I remembered the snappy ACDSee experience from 20 years ago, where the next image was preloading while you watched the current one. So, I built `biv` in Rust using GTK4 and libvips, with background decoding, bounded caching, and directional preloading.

The main design principle? **The next image should already be waiting.**

I wrote about the motivation and implementation here: [link]
