from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password, check_password
from .models import Fcuser
from .forms import LoginForm
import re
from ai.models import TempData2
import requests
import json
import os
import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import formatdate
from email import encoders


# Create your views here.
def send_mail(request):
    if request.method == 'GET':
        id = os.environ.get('PROJECT_EMAIL_ID')
        pw = os.environ.get('PROJECT_EMAIL_PW')
        email = os.environ.get('PROJECT_EMAIL_LINK')
        processing_mail(send_from=email, send_to=['dlalqnwk1081@naver.com', 'dlalqnwk1081@naver.com'],
                subject='인증파 이메일 전송!!', message=f'<h1>안녕하세요</h1>{id}입니다', files=[os.path.join('static/','등수.png')],
                mtype='html', server='smtp.naver.com', username=id, password=pw)
        
        return redirect('https://mail.naver.com')


def processing_mail(send_from, send_to, subject, message, mtype='plain', files=[],
              server="localhost", port=587, username='', password='',
              use_tls=True):
    
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = ', '.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message, mtype))

    for path in files:
        part = MIMEBase('application', "octet-stream")
        with open(path, 'rb') as file:
            part.set_payload(file.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition',
                        'attachment', filename=Path(path).name)
        msg.attach(part)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()


def home(request):
    return render(request, 'home.html')


def passwordchange(request):
    if request.method == 'POST':
        password = request.POST.get('password', None)
        re_password = request.POST.get('re-password', None)
        password = str(password)
        re_password = str(re_password)

        res_data = {}

        if not (password and re_password):
            res_data['error'] = '모든 값을 입력해야합니다.'
        elif password != re_password:
            res_data['error'] = '비밀번호가 다릅니다.'
        else:
            password_valid = check(password)
            temp = Fcuser.objects.filter(id=request.session['user']).last()
            if password_valid == True:
                if not check_password(password, temp.password):
                    temp.password = make_password(password)
                    temp.save()

                    return redirect('/')
                else:
                    res_data['error'] = '이전 비밀번호와 같습니다.'
            elif password_valid == False:
                if password == temp.username:
                    res_data['error'] = '비밀번호가 사용자 이름과 같습니다.'
                else:
                    res_data['error'] = '9자리 이상 숫자, 특수문자, 소문자, 대문자로 해주세요.'

        return render(request, 'passwordchange.html', res_data)
    else:
        return render(request, 'passwordchange.html')


def logout(request):
    if request.session.get('user'):
        user_id = request.session.get('user')
        fcuser = Fcuser.objects.get(pk=user_id)
        tempdata2 = TempData2.objects.filter(motion_writer=fcuser).last()
        if tempdata2 != None:
            tempdata2.session_num = 0
            tempdata2.motion_writer = fcuser
            tempdata2.save()

        elif tempdata2 == None:
            tempdata2 = TempData2()
            tempdata2.session_num = 0
            tempdata2.motion_writer = fcuser
            tempdata2.save()
        del (request.session['user'])
        del (request.session['exist'])
        
        data = {
            'grant_type': 'authorization_code',
            'client_id': os.environ.get('PROJECT_KEY'),
            'redirect_uri': os.environ.get('PROJECT_REDIRECT_URL'),
            'code': os.environ.get('PROJECT_CODE'),
        }

        response = requests.post(os.environ.get('PROJECT_URL'), data=data)
        tokens = response.json()

        url = os.environ.get('PROJECT_CONNECT')

        headers = {
            "Authorization": "Bearer " + tokens["access_token"]
        }

        data = {
            "template_object": json.dumps({
                "object_type": "text",
                "text": f'{fcuser} 로그아웃 했습니다.',
                "link": {
                    "web_url": "www.naver.com"
                }
            })
        }

        response = requests.post(url, headers=headers, data=data)
        response.status_code

    return redirect('/')


def login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            fcuser = Fcuser.objects.get(pk=form.user_id)

            if fcuser.login_judge == "success":
                request.session['user'] = form.user_id
                request.session['exist'] = 0
                tempdata2 = TempData2.objects.filter(
                    motion_writer=fcuser).last()
                if tempdata2 != None:
                    tempdata2.session_num = 0
                    tempdata2.motion_writer = fcuser
                    tempdata2.save()

                elif tempdata2 == None:
                    tempdata2 = TempData2()
                    tempdata2.session_num = 0
                    tempdata2.motion_writer = fcuser
                    tempdata2.save()

                data = {
                    'grant_type': 'authorization_code',
                    'client_id': os.environ.get('PROJECT_KEY'),
                    'redirect_uri': os.environ.get('PROJECT_REDIRECT_URL'),
                    'code': os.environ.get('PROJECT_CODE'),
                }

                response = requests.post(
                    os.environ.get('PROJECT_URL'), data=data)
                tokens = response.json()

                url = os.environ.get('PROJECT_CONNECT')

                headers = {
                    "Authorization": "Bearer " + tokens["access_token"]
                }

                data = {
                    "template_object": json.dumps({
                        "object_type": "text",
                        "text": f'{fcuser} 로그인 했습니다.',
                        "link": {
                            "web_url": "www.naver.com"
                        }
                    })
                }

                response = requests.post(url, headers=headers, data=data)
                response.status_code

                return redirect('/')
            elif fcuser.login_judge == "fail":
                form = LoginForm(request.POST)
                data = {
                    'grant_type': 'authorization_code',
                    'client_id': os.environ.get('PROJECT_KEY'),
                    'redirect_uri': os.environ.get('PROJECT_REDIRECT_URL'),
                    'code': os.environ.get('PROJECT_CODE'),
                }

                response = requests.post(
                    os.environ.get('PROJECT_URL'), data=data)
                tokens = response.json()

                url = os.environ.get('PROJECT_CONNECT')

                headers = {
                    "Authorization": "Bearer " + tokens["access_token"]
                }

                data = {
                    "template_object": json.dumps({
                        "object_type": "text",
                        "text": f'{fcuser}가 비밀번호 3회 틀려 잠겼습니다.',
                        "link": {
                            "web_url": "www.naver.com"
                        }
                    })
                }

                response = requests.post(url, headers=headers, data=data)
                response.status_code

                return render(request, 'login.html', {'form': form})
    else:
        form = LoginForm()

    return render(request, 'login.html', {'form': form})


def check(password):
    PT = []
    PT.append(re.compile('^(?=.*[A-Z])(?=.*[a-z])[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile('^(?=.*[A-Z])(?=.*\d)[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile(
        '^(?=.*[A-Z])(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{9,}$'))

    PT.append(re.compile('^(?=.*[a-z])(?=.*[A-Z])[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile('^(?=.*[a-z])(?=.*\d)[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile(
        '^(?=.*[a-z])(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{9,}$'))

    PT.append(re.compile('^(?=.*\d)(?=.*[A-Z])[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile('^(?=.*\d)(?=.*[a-z])[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile('^(?=.*\d)(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{9,}$'))

    PT.append(re.compile('^(?=.*[!@#$%^&*])(?=.*\d)[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile(
        '^(?=.*[!@#$%^&*])(?=.*[a-z])[A-Za-z\d!@#$%^&*]{9,}$'))
    PT.append(re.compile(
        '^(?=.*[!@#$%^&*])(?=.*[A-Z])[A-Za-z\d!@#$%^&*]{9,}$'))

    for num in range(len(PT)):
        if re.match(PT[num], password):
            return True
    return False


def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')
    elif request.method == 'POST':
        username = request.POST.get('username', None)
        useremail = request.POST.get('useremail', None)
        password = request.POST.get('password', None)
        re_password = request.POST.get('re-password', None)

        res_data = {}

        if not (username and useremail and password and re_password):
            res_data['error'] = '모든 값을 입력해야합니다.'
        elif password != re_password:
            res_data['error'] = '비밀번호가 다릅니다.'
        else:
            password_valid = check(password)
            if password_valid == True:
                if password != username:
                    username = str(username)
                    password = str(password)

                    temp = Fcuser.objects.filter(username=username)
                    if temp:
                        res_data['error'] = '다시 생성해주세요.'
                        return render(request, 'register.html', res_data)

                    temp = Fcuser.objects.filter(useremail=useremail)
                    if temp:
                        res_data['error'] = '다시 생성해주세요.'
                        return render(request, 'register.html', res_data)

                    for word in ['|', ';', '&', ':', '>', '<', '`', '\\', '!', '/']:
                        if word in username:
                            res_data['error'] = '|, ;, &, :, >, <, `, \, !, / 을 제외한 문자로 username을 만들어주세요.'

                            return render(request, 'register.html', res_data)

                    fcuser = Fcuser(
                        username=username,
                        useremail=useremail,
                        password=make_password(password),
                    )
                    fcuser.save()
                    return redirect('/')

                elif password == username:
                    res_data['error'] = '비밀번호가 사용자 이름과 같습니다.'
            elif password_valid == False:
                if password == username:
                    res_data['error'] = '비밀번호가 사용자 이름과 같습니다.'
                else:
                    res_data['error'] = '9자리 이상 숫자, 특수문자, 소문자, 대문자로 해주세요.'

        return render(request, 'register.html', res_data)
