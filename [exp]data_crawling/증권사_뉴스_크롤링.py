from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os 
import time
 
os.chdir('./')
 
list_len_zero_company = []
 
def crawler(company, company_code, maxpage):
    
    page = 1 

    save_company_news_data = "./save_news/%s"%company
    if os.path.exists(save_company_news_data):
        print("이미 크롤링을 완료 하였습니다.")
        return
    else:
        os.makedirs(save_company_news_data)

    f = open("%s/%s_title.txt"%(save_company_news_data, company), 'w')
    print(company, company_code)

    while page <= int(maxpage): 
    
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page) 
        source_code = requests.get(url).text
        html = BeautifulSoup(source_code, "lxml")
     
 
        
        # 뉴스 제목 
        titles = html.select('.title')
        title_result=[]
        for title in titles: 
            title = title.get_text() 
            title = re.sub('\n','',title)
            title_result.append(title)
 
 
        # 뉴스 링크
        links = html.select('.title') 
 
        link_result =[]
        for link in links: 
            add = 'https://finance.naver.com' + link.find('a')['href']
            link_result.append(add)
 
 
        # 뉴스 날짜 
        dates = html.select('.date') 
        date_result = [date.get_text() for date in dates] 
 
 
        # 뉴스 매체     
        sources = html.select('.info')
        source_result = [source.get_text() for source in sources] 
 
 
        # 변수들 합쳐서 해당 디렉토리에 csv파일로 저장하기 
 
        result= {"날짜" : date_result, "언론사" : source_result, "기사제목" : title_result, "링크" : link_result} 
        df_result = pd.DataFrame(result)
        
        print("다운 받고 있습니다------")
        # print()
        df_result.to_csv(save_company_news_data + '/page' + str(page) + '.csv', mode='w', encoding='utf-8-sig') 

        # f1 = open("%s/%s_article.txt"%(save_company_news_data, company), "w")
        print(link_result)
        for link in link_result:
            time.sleep(0.5)
            r = requests.get(link)

            soup = BeautifulSoup(r.text, "lxml")
            news = soup.find('div', id='content')
            try:
                f.write("%s\n"%str(news.find('strong', class_="c p15").text))
            except:
                pass

            # for tag in news.find_all(['a']): # give the list of tags you want to ignore here.
            #     tag.replace_with('')

            # f1.write("%s"%str(news.find('div', id="news_read", class_="scr01").text))

        page += 1 
    if len(link_result) == 0 :
        list_len_zero_company.append(company)

    f.close()
    f1 = open("%s/%s_title.txt"%(save_company_news_data, company), 'r')
    tmp = f1.read()
    print(len(tmp))
    print("%s 크롤링이 완료되었습니다. \n"%company)
 
 
    
 
# 종목 리스트 파일 열기  
# 회사명을 종목코드로 변환 
        
def convert_to_code(maxpage = 2):
    path = "/Users/samsung/Downloads/kospi.xlsx"

    df = pd.read_excel(path)
    # 2020년 7월 24일 기준
    # 코스피 회사 : 791개, 업종 : 126개
    kospi_company_list = df['회사명']
    df['종목코드'] = df['종목코드'].astype(str)
    kospi_code_list = df['종목코드']
    # df['A'].astype(str)
    kospi_category_list = df['업종']

    tmp = df.drop_duplicates('업종')


    keys = [i for i in kospi_company_list]    #데이터프레임에서 리스트로 바꾸기 
 
    company_code = df['종목코드']
    values = [j for j in company_code]
 
    dict_result = dict(zip(keys, values))  # 딕셔너리 형태로 회사이름과 종목코드 묶기 
    
    pattern = '[a-zA-Z가-힣]+' 
    
 
    for company_name, company_code in zip(kospi_company_list, kospi_code_list):
        # zfill(원하는 자리수) -> 0으로 앞을 채워줌
        company_code = company_code.zfill(6)
        crawler(company_name, company_code, maxpage)

    f2 = open("./len_zero_company", "w")
    f2.write(list_len_zero_company)

    f2.close()
        
           
 
 
def main():
    info_main = input("="*50+"\n"+"실시간 뉴스기사 다운받기."+"\n"+" 시작하시려면 Enter를 눌러주세요."+"\n"+"="*50)
    
    if os.path.exists("./save_news"):
        pass
    else:
        os.makedirs("./save_news")
 
    convert_to_code() 
 


 
main() 