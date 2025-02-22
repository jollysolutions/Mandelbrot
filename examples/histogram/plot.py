from PIL import Image, ImageDraw
from mandelbrot import mandelbrot, MAX_ITER
from collections import defaultdict
from math import floor, ceil
from pprint import pp

def linear_interpolation(color1, color2, t):
    return color1 * (1 - t) + color2 * t

# Image size (pixels)
WIDTH = 600
HEIGHT = 600

# Plot window
RE_START = -2
RE_END = 0.5
IM_START = -1.25
IM_END = 1.25

histogram = defaultdict(lambda: 0)
values = {}
values2 = {}
for x in range(0, WIDTH):
    for y in range(0, HEIGHT):
        # Convert pixel coordinate to complex number
        c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                    IM_START + (y / HEIGHT) * (IM_END - IM_START))
        # Compute the number of iterations
        m = mandelbrot(c)

        values[(x, y)] = m
        values2[(x, y)] = m/MAX_ITER
        if m < MAX_ITER:
            histogram[floor(m)] += 1

total = sum(histogram.values())
hues = []
h = 0
for i in range(MAX_ITER):
    h += histogram[i] / total
    hues.append(h)
hues.append(h)

im = Image.new('HSV', (WIDTH, HEIGHT), (0, 0, 0))
draw = ImageDraw.Draw(im)
h_v = {}
for x in range(0, WIDTH):
    for y in range(0, HEIGHT):
        m = values[(x, y)]
        # The color depends on the number of iterations
        hue = 255 - int(255 * linear_interpolation(hues[floor(m)], hues[ceil(m)], m % 1))
        h_v[(x,y)] = hue/255
        saturation = 255
        value = 255 if m < MAX_ITER else 0
        # Plot the point
        draw.point([x, y], (hue, saturation, value))

#im.convert('RGB').save('output.png', 'PNG')
im.convert('RGB').show()
