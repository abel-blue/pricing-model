import json
import pandas as pd
import numpy as np
from zipfile import ZipFile
from itertools import zip_longest

def read_json(json_file: str) -> list:
    # json_data = []
    with ZipFile(json_file, 'r') as zip_ref:
        zip_ref.extractall("data/")
        json_data = json.load(open("data/global_design_data.json", 'r'))
        dataa = json_data.items()
        listt = list(dataa)
    return len(listt), listt
# 
class CmpDfExtractor:

    def __init__(self, cmp_list):
        self.cmp_list = cmp_list
        print('Data Extraction in progress...')
    
    def gamekey_extractor(self):
        game_key = []
        for x in self.cmp_list:
            for key in x[1]:
                game_key.append(x[0] + "/" + key)
        return game_key
    
    def labels_extractor(self):
        labels_engagement = []
        labels_clickthr = []
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'labels' in v.keys():
                    labels_engagement.append(",".join(v['labels']['engagement']))
                    labels_clickthr.append(",".join(v['labels']['click_through']))
                else:
                    labels_engagement.append(None)
                    labels_clickthr.append(None)
        return labels_engagement, labels_clickthr
        
    def text_extractor(self):
        text_engagement = []
        text_clickthr = []
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'text' in v.keys():
                    text_engagement.append(",".join(v['text']['engagement']))
                    text_clickthr.append(",".join(v['text']['click_through']))
                else:
                    text_engagement.append(None)
                    text_clickthr.append(None)
        return text_engagement, text_clickthr
        
    def color_extractor(self):
        colors_engagement_red = []
        colors_engagement_green = []
        colors_engagement_blue = []
        colors_engagement_proportion = []
        colors_engagement_saturation = []
        colors_engagement_luminosity = []
        # 
        colors_clickthr_red = []
        colors_clickthr_green = []
        colors_clickthr_blue  = []
        colors_clickthr_proportion = []
        colors_clickthr_saturation = []
        colors_clickthr_luminosity = []
        # 
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'colors' in v.keys():
                    if "1" in v['colors']['engagement'].keys():
                        colors_engagement_red.append(v['colors']['engagement']['1']['red'])
                        colors_engagement_green.append(v['colors']['engagement']['1']['green'])
                        colors_engagement_blue.append(v['colors']['engagement']['1']['blue'])
                        colors_engagement_proportion.append(v['colors']['engagement']['1']['proportion'])
                        colors_engagement_saturation.append(v['colors']['engagement']['1']['saturation'])
                        colors_engagement_luminosity.append(v['colors']['engagement']['1']['luminosity'])
                    else:
                        colors_engagement_red.append(None)
                        colors_engagement_green.append(None)
                        colors_engagement_blue.append(None)
                        colors_engagement_proportion.append(None)
                        colors_engagement_saturation.append(None)
                        colors_engagement_luminosity.append(None)
                        # 
                    if "1" in v['colors']['click_through'].keys():
                        colors_clickthr_red.append(v['colors']['click_through']['1']['red'])
                        colors_clickthr_green.append(v['colors']['click_through']['1']['green'])
                        colors_clickthr_blue.append(v['colors']['click_through']['1']['blue'])
                        colors_clickthr_proportion.append(v['colors']['click_through']['1']['proportion'])
                        colors_clickthr_saturation.append(v['colors']['click_through']['1']['saturation'])
                        colors_clickthr_luminosity.append(v['colors']['click_through']['1']['luminosity'])
                    else: 
                        colors_clickthr_red.append(None)
                        colors_clickthr_green.append(None)
                        colors_clickthr_blue.append(None)
                        colors_clickthr_proportion.append(None)
                        colors_clickthr_saturation.append(None)
                        colors_clickthr_luminosity.append(None)
            
        return colors_engagement_red, colors_engagement_green, colors_engagement_blue, colors_engagement_proportion, colors_engagement_saturation, colors_engagement_luminosity, colors_clickthr_red, colors_clickthr_green, colors_clickthr_blue, colors_clickthr_proportion, colors_clickthr_saturation, colors_clickthr_luminosity
    
    def videosd_extractor(self):
        videos_data = []
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'videos_data' in v.keys():
                    videos_data.append(v['videos_data']['has_video'])
        return videos_data
    
    def eng_type_extractor(self):
        eng_type_data = []
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'eng_type' in v.keys():
                    eng_type_data.append(v['eng_type']['eng_type'])
                else:
                    eng_type_data.append(None)
        return eng_type_data
    
    def direction_extractor(self):
        direction_data = []
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'direction' in v.keys():
                    direction_data.append(v['direction']['direction'])
                else:
                    direction_data.append(None)
        return direction_data
    
    def adunit_size_extractor(self):
        adunit_sizex = []
        adunit_sizey = []
        for x in self.cmp_list:
            for k, v in x[1].items():
                if 'adunit_sizes' in v.keys():
                    adunit_sizex.append(v['adunit_sizes']['size_x'])
                    adunit_sizey.append(v['adunit_sizes']['size_y'])
                else:
                    adunit_sizex.append(None)
                    adunit_sizey.append(None)
        return adunit_sizex, adunit_sizey
        
        
    def get_cmp_df(self, save=False) -> pd.DataFrame:

        columns = ['game_key', 'labels_engagement', 'labels_clickthr', 
                   'text_engagement', 'text_clickthr', 'colors_engagement_red', 
                   'colors_engagement_green', 'colors_engagement_blue', 
                   'colors_engagement_proportion', 'colors_engagement_saturation', 
                   'colors_engagement_luminosity', 'colors_clickthr_red', 
                   'colors_clickthr_green', 'colors_clickthr_blue', 'colors_clickthr_proportion', 
                   'colors_clickthr_saturation', 'colors_clickthr_luminosity',
                   'videosd', 'eng_type', 'direction', 'adunit_sizex', 'adunit_sizey']

        game_key = self.gamekey_extractor()
        labels_engagement, labels_clickthr = self.labels_extractor()
        text_engagement, text_clickthr = self.text_extractor() 
        colors_engagement_red, colors_engagement_green, colors_engagement_blue, colors_engagement_proportion, colors_engagement_saturation, colors_engagement_luminosity, colors_clickthr_red, colors_clickthr_green, colors_clickthr_blue, colors_clickthr_proportion, colors_clickthr_saturation, colors_clickthr_luminosity = self.color_extractor()
        videosd = self.videosd_extractor()
        eng_type = self.eng_type_extractor()
        direction = self.direction_extractor()
        adunit_sizex, adunit_sizey = self.adunit_size_extractor()
        
        data = zip(game_key, labels_engagement, labels_clickthr, text_engagement, 
                           text_clickthr, colors_engagement_red, colors_engagement_green, colors_engagement_blue, colors_engagement_proportion, colors_engagement_saturation, colors_engagement_luminosity, colors_clickthr_red, colors_clickthr_green, colors_clickthr_blue, colors_clickthr_proportion, colors_clickthr_saturation, colors_clickthr_luminosity,
                           videosd, eng_type, direction, adunit_sizex, adunit_sizey)
        df = pd.DataFrame(data=data, columns=columns)

        if save:
            df.to_csv('data/extracted_cmp_data.csv', index=False)
            print('Extracted Data Saved !!!')
        return df

if __name__ == "__main__":
    _, cmp_list = read_json("data/global_design_data.zip")
    cmp = CmpDfExtractor(cmp_list)
    df = cmp.get_cmp_df(True)