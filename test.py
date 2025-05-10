import requests

url = "https://query1.finance.yahoo.com/v8/finance/chart/TSM?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1d&period1=852094800&period2=1746852604&symbol=TSM&userYfid=true&lang=en-US&region=US"

"""
const requestOptions = {
    method: "GET",
    redirect: "follow"
  };
  
  fetch("https://query1.finance.yahoo.com/v8/finance/chart/TSM?events=capitalGain%7Cdiv%7Csplit&formatted=true&includeAdjustedClose=true&interval=1d&period1=852094800&period2=1746852604&symbol=TSM&userYfid=true&lang=en-US&region=US", requestOptions)
    .then((response) => response.text())
    .then((result) => console.log(result))
    .catch((error) => console.error(error));
"""
cookies = "GUC=AQEBCAFoFwZoREIeHwSS&s=AQAAABu-hjtc&g=aBW_KA; A1=d=AQABBA7jomcCENLxvbr8wM102FcNiWJuZhEFEgEBCAEGF2hEaC5YyCMA_eMDAAcIDuOiZ2JuZhE&S=AQAAAhs8Jx5Z5we9WjY9Hq8_F94; A3=d=AQABBA7jomcCENLxvbr8wM102FcNiWJuZhEFEgEBCAEGF2hEaC5YyCMA_eMDAAcIDuOiZ2JuZhE&S=AQAAAhs8Jx5Z5we9WjY9Hq8_F94; A1S=d=AQABBA7jomcCENLxvbr8wM102FcNiWJuZhEFEgEBCAEGF2hEaC5YyCMA_eMDAAcIDuOiZ2JuZhE&S=AQAAAhs8Jx5Z5we9WjY9Hq8_F94; DSS=sdtp=mcafee&sdts=1746849583&ts=1739768606&cnt=0"
# add headers   
headers = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "es-ES,es;q=0.9",
    "priority": "u=0, i",
    "sec-ch-ua": "\"Chromium\";v=\"136\", \"Google Chrome\";v=\"136\", \"Not.A/Brand\";v=\"99\"",
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "\"Windows\"",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "none",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "cookie": cookies,
    "user-agent": "Mozilla/5.0",
  }
response = requests.get(url, headers=headers)
print(response.text)

