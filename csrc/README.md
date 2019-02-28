# Embedded Image Processing  
This is directory contains code intended to be run on device for implementing basic image quality checks. Purpose of this code is to filter out images that do not satisfy certain criteria on device itself.  
Multiple criteria that we can think of are so far:  
- Invalid JPEG encoding
- White-out/black-out 
- Blurring (motion/focus)
- Images not containing face

## JPEG Decoder
In our standard flow, we use JPEG encoded images for reducing bandwidth usage and memory limitations. It's not practical to obtain raw image from camera and then write encoder on device to convert it to JPEG encoded format or send raw image to server. Thus we need to write JPEG parser that down samples the original image which can be processed on device for applying various image filters.
