from selenium import webdriver
import bs4 as bs
from urllib.request import Request, urlopen
import pandas as pd
import time
import os
import requests
from IPython import embed

# os.chdir('/data')

website = 'https://www.thecarconnection.com'


def fetch(page, addition=''):
    return bs.BeautifulSoup(urlopen(Request(page + addition,
            headers={'User-Agent': 'Opera/9.80 (X11; Linux i686; Ub'\
                     'untu/14.10) Presto/2.12.388 Version/12.16'})).read(), 'lxml')

def all_makes():
    # Fetches all makes (acura, cadilac, etc)
    all_makes_list = []
    for a in fetch(website, "/new-cars").find_all("a", {"class": "add-zip"}):
        all_makes_list.append(a['href'])
    print(all_makes_list[:10])
    print("All makes fetched")
    return all_makes_list


def make_menu(listed):
    # Fetches all makes + model ? (acura_mdx, audi_q3, etc)
    make_menu_list = []
    for make in listed: # REMOVE REMOVE REMOVE REMOVE REMOVE REMOVE #
        for div in fetch(website, make).find_all("div", {"class": "name"}):
            make_menu_list.append(div.find_all("a")[0]['href'])
    print(make_menu_list[:10])
    print("Make menu list fetched")
    return make_menu_list


def model_menu(listed):
    # Add year to previous step
    model_menu_list = []
    for make in listed:
        soup = fetch(website, make)
        for div in soup.find_all("a", {"class": "btn avail-now first-item"}):
            model_menu_list.append(div['href'])
        for div in soup.find_all("a", {"class": "btn 1"})[:8]:
            model_menu_list.append(div['href'])
    print(model_menu_list[:10])
    print("Model menu list fetched")
    return model_menu_list


def year_model_overview(listed):
    year_model_overview_list = []
    for make in listed: # REMOVE REMOVE REMOVE REMOVE REMOVE REMOVE REMOVE REMOVE
        for id in fetch(website, make).find_all("a", {"id": "ymm-nav-specs-btn"}):
            year_model_overview_list.append(id['href'])
    try:
        year_model_overview_list.remove("/specifications/buick_enclave_2019_fwd-4dr-preferred")
    except:
        pass
    print(year_model_overview_list[:10])
    print("Year model overview list fetched")
    return year_model_overview_list


def trims(listed):
    trim_list = []
    for row in listed:
        div = fetch(website, row).find_all("div", {"class": "block-inner"})[-1]
        div_a = div.find_all("a")
        for i in range(len(div_a)):
            trim_list.append(div_a[-i]['href'])
    print(trim_list[:10])
    print("Trims list fetched")
    return trim_list


def timer(start, end, iters, iters_left):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)

    hours_per_iter, rem_per_iter = divmod((end-start)/(iters+1),3600)
    minutes_per_iter, seconds_per_iter = divmod(rem_per_iter,60)

    hours_left , rem_left = divmod(((end-start)/(iters+1))*iters_left,3600)
    minutes_left, seconds_left = divmod(rem_left,60)
    print("    Total elapsed: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    print("    Time per page: {:0>2}:{:0>2}:{:05.2f}".format(int(hours_per_iter),int(minutes_per_iter),seconds_per_iter))
    print("    Time left    : {:0>2}:{:0>2}:{:05.2f}".format(int(hours_left),int(minutes_left),seconds_left))


def saveImage(imgUrl, imgName, group):
    imgData = requests.get(imgUrl).content
    with open('data/pictures/'+group+'/'+imgName+'.jpg','wb') as handler:
        handler.write(imgData)


def create_file_name(row):
    ''' 
        Takes all columns not named pictures and delimits it with --
        Also replaces spaces in individual columns with _
    '''

    # rownames = [str(x).strip().replace(',','').replace(' ','_').replace('/','.') for inx, x in row.iteritems()[0]
    #                                      if 'Picture_' not in inx]
    # return '--'.join(rownames)
    # print(row[0])
    return row[0].strip().replace(' ','_').replace('/','.')


def specifications(website, trims, keep_all_images=True):
    ''' keep_all_images: True means we create 2 files, one for main (front/read)
                         And one for all of the pictures.
    '''
    options = webdriver.FirefoxOptions()
    options.add_argument('-headless')
    driver = webdriver.Firefox(options=options)
    # driver = webdriver.Firefox()

    # Timer start
    start = time.time()

    if not os.path.isfile('data/pictures_all.csv'):
        # Table for all images
        specifications_table_all = pd.DataFrame()
        # Table for only front and rear images
        specifications_table_front_rear = pd.DataFrame()
    else:
        specifications_table_all = pd.read_csv('data/pictures_all.csv',index_col=0)
        specifications_table_front_rear = pd.read_csv('data/pictures_rear_front.csv',index_col=0)

    trims_left = len(trims.index)
    if trims_left == 0:
        return 0
    for inx, webpage in enumerate(trims.iloc[:, 0]):
        soup = fetch(website, webpage.replace('overview', 'specifications'))
        # Same splitting as above
        specifications_df_all = pd.DataFrame(columns=[soup.find_all("title")[0].text[:-15]])
        specifications_df_front_rear = pd.DataFrame(columns=[soup.find_all("title")[0].text[:-15]])
        for div in soup.find_all("div", {"class": "specs-set-item"})[:9]:
            row_name = div.find_all("span")[0].text
            row_value = div.find_all("span")[1].text
            specifications_df_all.loc[row_name] = row_value
            specifications_df_front_rear.loc[row_name] = row_value
        
        try:
            driver.get(website + webpage.replace('overview', 'photos'))
            time.sleep(0.5)
            ext_btn = driver.find_element_by_class_name('view-mode.show-ext')
            if ext_btn.text == 'Exterior':
                ext_btn.click()
            time.sleep(0.5)
            class_img_ext = driver.find_elements_by_xpath("//div[@class='thumbs-wrapper']/div[starts-with(@class, 'thumbs-slide') and not(contains(@class, 'video'))]/img")
            list_urls = [x.get_attribute("src").replace('/tmb/','/sml/').replace('_t.gif','_s.jpg') for x in class_img_ext]
        except:
            list_urls = []
            print(f'Problem with {website + webpage}')
        
        # Different layout for older images
        # if len(class_img) == 0:
        #     try:
        #         driver.get(website + webpage.replace('overview', 'photos'))
        #         class_img = driver.find_elements_by_class_name('image')
        #         list_urls = []
        #         for ii in class_img:
        #             list_urls.append(ii.get_attribute('data-image-small'))
        #     except:
        #         print(f'Problem with {website + webpage}')
        

        
        # Keep a count of rear and front images to put them at start of index
        rear_front_img_count = 0
        for ix, img_url in enumerate(list_urls): # REMOVE REMOVE REMOVE
            specifications_df_all.loc['Picture_%i' % ix, :] = img_url
            if keep_all_images and 'pkg-rear-exterior-view' in img_url:
                specifications_df_front_rear.loc['Picture_%i' % rear_front_img_count, :] = img_url
                rear_front_img_count += 1
            
        # If no images, we don't add to the main df
        if len(list_urls) > 0:
            specifications_table_all = pd.concat([specifications_table_all, specifications_df_all], axis=1, sort=False)
            if rear_front_img_count > 0:
                specifications_table_front_rear = pd.concat([specifications_table_front_rear, specifications_df_front_rear], axis=1, sort=False)

        # Save content every 10 images
        if inx % 10 == 0:
            print("%d/%d completed."%(inx, trims_left))
            specifications_table_all.to_csv('data/pictures_all.csv')
            specifications_table_front_rear.to_csv('data/pictures_rear_front.csv')
            trims.iloc[inx:].to_csv('data/trims_octobre_2019.csv', header=None)
            timer(start,time.time(), inx, trims_left-inx)


    # At the end of loop
    specifications_table_all.to_csv('data/pictures_all.csv')
    specifications_table_front_rear.to_csv('data/pictures_rear_front.csv')
    specifications_table_all.to_csv('data/img_left_octobre_2019.csv')
    specifications_table_front_rear.to_csv('data/img_left_frontrear_octobre_2019.csv')

fetch_urls       = False
download_images  = True

if __name__ == '__main__':
    if fetch_urls:
        # If list of trims has not been fetched
        if not os.path.isfile('data/trims_octobre_2019.csv'):
            a = all_makes()
            b = make_menu(a)
            c = model_menu(b)
            d = year_model_overview(c)
            e = trims(d)
            f = pd.DataFrame(e).to_csv('data/trims_octobre_2019.csv', header=None)
            # Previous one will be modified
            f = pd.DataFrame(e).to_csv('data/trims_octobre_2019_keep.csv', header=None)

        # Read list of trims
        g = pd.read_csv('data/trims_octobre_2019.csv',index_col=0, header=None)
        g.drop_duplicates(inplace=True)
        h = specifications(website, g)

    if download_images:
        i_all = pd.read_csv('data/img_left_octobre_2019.csv',index_col=0)
        i_front_rear = pd.read_csv('data/img_left_frontrear_octobre_2019.csv',index_col=0)

        i_all = i_all.transpose().reset_index()
        i_front_rear = i_front_rear.transpose().reset_index()

        i_all['imgName'] = i_all.apply(create_file_name, axis=1)
        i_front_rear['imgName'] = i_front_rear.apply(create_file_name, axis=1)

        start = time.time()
        
        for ind, row in i_all.iterrows():
            if ind % 10 == 0:
                timer(start, time.time(), ind, len(i_all.index))
                print('%i/%i image pages for all angles completed.' %(ind,len(i_all.index)))
                i_all.iloc[ind:].to_csv('data/img_left_octobre_2019.csv')
            img_urls = [x for inx, x in row.iteritems() if 'Picture_' in inx and str(x) != 'nan']
            pic_name = row['imgName']
            for ix, url in enumerate(img_urls):
                saveImage(url, pic_name+'_'+str(ix), 'all_images')

        start = time.time()
        for ind, row_front in i_front_rear.iterrows():
            if ind % 10 == 0:
                timer(start, time.time(), ind, len(i_front_rear.index))
                print('%i/%i image pages for front/rear completed.' %(ind,len(i_front_rear.index)))
                i_front_rear.iloc[ind:].to_csv('data/img_left_frontrear_octobre_2019.csv')
            img_urls = [x for inx, x in row_front.iteritems() if 'Picture_' in inx and str(x) != 'nan']
            pic_name = row_front['imgName']
            for ix, url in enumerate(img_urls):
                saveImage(url, pic_name+'_'+str(ix), 'front_rear')

    











