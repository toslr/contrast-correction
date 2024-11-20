import skimage
import numpy as np

in_path = '/Users/tom/Downloads/D-8000X-CC-1_preblend-2,1.tif'
out_path = '/Users/tom/Downloads/D-8000X-CC-1_preblend-2,1_norm.tif'
in_path = '/Users/tom/Downloads/D-8000X-CC-1_preblend-2-1_8bit_eq_resized.png'
out_path = '/Users/tom/Downloads/D-8000X-CC-1_preblend-2-1_8bit_eq_resized_norm10.png'

n = 10 # number of tiles per side (n^2 tiles total). less tiles is harder to handle for CPU in histogram equalization.
clip_limit = 0.01 #clip between 0.01 and 0.05. Higher = more contrast

def crop_image_to_squares(image, n):
    """Cut the image in n*n tiles and return the list of the tiles."""
    width, height = image.shape
    if width != height:
        raise ValueError("Image must be square")
    size_of_square = width // n
    
    cropped_images = []
    for i in range(n):
        for j in range(n):
            left = j * size_of_square
            top = i * size_of_square
            right = left + size_of_square
            bottom = top + size_of_square
            cropped_img = image[top:bottom, left:right]
            cropped_images.append(cropped_img)
    return cropped_images

def main():
    img = skimage.io.imread(in_path,as_gray=True)
    tiles = crop_image_to_squares(img, n)
    ref_tile = tiles[n*n//2+n//2] #define ref tile as a middle one
    ref_tile = skimage.exposure.equalize_adapthist(ref_tile,clip_limit=clip_limit) 
    matched_tiles = []
    for i,tile in enumerate(tiles):
        matched_tile = skimage.exposure.equalize_adapthist(tile, clip_limit = clip_limit) #local contrast enhancement on each tile
        matched_tile = skimage.exposure.match_histograms(matched_tile, ref_tile) #match histogram to model_tile
        matched_tile = skimage.img_as_ubyte(matched_tile) #ensure 8bit img
        matched_tiles.append(matched_tile)

    # Stitch the matched tiles back together and save
    stitched_image = np.concatenate([np.concatenate([matched_tiles[j + i * n] for j in range(n)], axis=1) for i in range(n)], axis=0)
    skimage.io.imsave(out_path,stitched_image)

if __name__ == '__main__':
    main()