import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
#1/20 수정

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'] #xml파일을 csv 파일로 변환 후 column(속성) 멤버들
    xml_df = pd.DataFrame(xml_list, columns=column_name) # pandas dataframe -> 비프자료 참고(표 형식)
    # #pandas의 DataFrame.to_csv() 함수를 사용
    return xml_df # DataFrame 형태의 변수반환


def main():
    flag = 0
    root = os.path.join(os.getcwd(), 'images')
    while True: # train과 val폴더 둘 다 csv 파일을 생성해냄 각각 별도로
        if os.listdir('images\\train') and flag == 0: # C:\Users\jinwo\models\research\object_detection\ 에서 하위 dir에 train 있는지 여부, train이 있으면 1반환
            print("train folder catched")
            image_path = os.path.join(os.getcwd(), 'images\\train') #os.getcwd() 현재경로에서 join # annotations-> image폴더로 변경
            xml_df = xml_to_csv(image_path) # xml_df는 pandas의 DataFrame 형태를 받음
            filename = 'train_'
            xml_df.to_csv(root + "\\" + filename + 'labels.csv', index=None) #생성된 csv 파일 저장
            flag = 2
        elif os.listdir('images\\val') and flag == 2:
            print("val folder catched")
            image_path = os.path.join(os.getcwd(), 'images\\val')  # annotations-> image폴더로 변경 필요
            xml_df = xml_to_csv(image_path)
            filename = 'val_'
            #root = image_path.os.path.abspath('../')
            xml_df.to_csv(root + "\\" + filename + 'labels.csv', index=None) #생성될 root + csv명
            flag = 3
        elif flag == 3:
            break
        else:
            print("No folder in DIR")
            flag = 1
    #print(image_path)
    #time.sleep(10)
    if flag is not 1:
        print('Successfully converted xml to csv.')

main()

