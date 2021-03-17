# 네이버 뉴스의 카테고리 별로 뉴스 추출

from korea_news_crawler.articlecrawler import ArticleCrawler


def main():
    Crawler = ArticleCrawler()  
    Crawler.set_category('정치', '경제', '사회')  
    Crawler.set_date_range(2021, 1, 2021, 1)  
    Crawler.start()

if __name__ =="__main__":
    main()