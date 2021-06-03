import pymysql
from pandas import DataFrame


def connect_db():
    sharenote_db = pymysql.connect(
        user='root', 
        passwd='sharenotedev1!', 
        host='52.79.246.196', 
        port=3306,
        db='share_note', 
        charset='utf8'
    )

    # data read -> 오늘 기준으로 전날 데이터 조회하는 쿼리 필요
    cursor = sharenote_db.cursor(pymysql.cursors.DictCursor)

    sql = "SELECT * FROM `topic_news`;"
    cursor.execute(sql)
    result = cursor.fetchall()

    df = DataFrame(result)
    print(df['topic_news_title'])

    return df


if __name__ == "__main__":
    connect_db()