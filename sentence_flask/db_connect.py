import pymysql

db = pymysql.connect(
    user='root', 
    passwd='{설정한 비밀번호}', 
    host='', 
    db='juso-db', 
    charset='utf8'
)