from PIL import Image as image
import os
import argparse

#print(os.listdir(directory))
def rescale_images(directory, size): # 해당 디렉토리의 이미지 갯수 (size) 만큼 반복
    image_file = {'.jpg', '.png', '.bmp'}
    #print(os.listdir(directory))
    for img in os.listdir(directory):
        for i in image_file:
            if i in img:
                print('1')
                im = image.open(directory + img) #open image from directory
                im_resized = im.resize(size, image.ANTIALIAS) #image resize to size
                im_resized.save(directory + img) #image save

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rescale Images")
    parser.add_argument('-d', '--directory', type=str, required=True, help='Directory')
    parser.add_argument('-s', '--size', type=int, nargs=2, required = True, metavar=('width','height'))
    #metavar = size, size / parser.add_argument로 cmd 명령어 만들어주기 / resizing 전 설정부분
    args = parser.parse_args()
    print(args)
    print(args.directory)
    rescale_images(args.directory, args.size) #resize로 JUMP

