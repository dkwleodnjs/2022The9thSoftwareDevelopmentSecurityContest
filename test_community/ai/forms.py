from django import forms

class MotionAIForm(forms.Form):
    motion_name1 = forms.CharField(
        error_messages={
            'required': '모션1 이름을 입력해주세요.'
        },
        max_length=256, label="모션1 이름", initial='')
    motion_name2 = forms.CharField(
        error_messages={
            'required': '모션2 이름을 입력해주세요.'
        },
        max_length=256, label="모션2 이름", initial='')
    motion_name3 = forms.CharField(
        error_messages={
            'required': '모션3 이름을 입력해주세요.'
        },
        max_length=256, label="모션3 이름", initial='')
    motion_name4 = forms.CharField(
        error_messages={
            'required': '모션4 이름을 입력해주세요.'
        },
        max_length=256, label="모션4 이름", initial='')
    motion_name5 = forms.CharField(
        error_messages={
            'required': '모션5 이름을 입력해주세요.'
        },
        max_length=256, label="모션5 이름", initial='')
    motion_name6 = forms.CharField(
        error_messages={
            'required': '모션6 이름을 입력해주세요.'
        },
        max_length=256, label="모션1 이름", initial='')
