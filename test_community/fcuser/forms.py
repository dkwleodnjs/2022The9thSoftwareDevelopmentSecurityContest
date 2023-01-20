from django import forms
from .models import Fcuser
from django.contrib.auth.hashers import check_password


class LoginForm(forms.Form):
    username = forms.CharField(
        error_messages={
            'required': '아이디를 입력해주세요.'
        },
        max_length=128, label="사용자 이름")
    password = forms.CharField(
        error_messages={
            'required': '비밀번호를 입력해주세요.'
        },
        widget=forms.PasswordInput, max_length=128, label="비밀번호")

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        if username and password:
            username = str(username)
            password = str(password)

            try:
                fcuser = Fcuser.objects.get(username=username)
            except Fcuser.DoesNotExist:
                self.add_error('username', '다시 입력해주세요.')
                return

            fail = Fcuser.objects.get(username=username)
            if not check_password(password, fcuser.password):
                fail.fail_account_count += 1
                fail.save()

                if fail.login_judge == 'success':
                    if fail.fail_account_count >= 3:
                        fail.login_judge = 'fail'
                        self.user_login_judge = fail.login_judge
                        self.user_id = fcuser.id
                        fail.save()
                    else:
                        self.add_error('password', '다시 입력해주세요.')
                elif fail.login_judge == 'fail':
                    self.user_login_judge = fail.login_judge
                    self.user_id = fcuser.id
                    self.add_error('password', '계정이 잠겼습니다. 관리자한테 연락바랍니다.')

            elif fail.login_judge == 'fail':
                self.user_login_judge = fail.login_judge
                self.user_id = fcuser.id
                self.add_error('password', '계정이 잠겼습니다. 관리자한테 연락바랍니다.')

            elif fail.login_judge == 'success':
                fail.fail_account_count = 0
                fail.save()
                self.user_id = fcuser.id
