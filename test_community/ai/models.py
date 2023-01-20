from django.db import models

# Create your models here.


class MotionAI(models.Model):
    motion_name = models.CharField(max_length=256,
                                    verbose_name='모션 이름')
    prev_hash = models.BinaryField(max_length=256,
                                   verbose_name='이전 블록체인 데이터')
    current_hash = models.BinaryField(max_length=256,
                                      verbose_name='현재 블록체인 데이터')
    motion_writer = models.ForeignKey('fcuser.Fcuser', on_delete=models.CASCADE,
                                      verbose_name='작성자')
    registered_dttm = models.DateTimeField(auto_now_add=True,
                                           verbose_name='등록시간')
    make_data_count = models.IntegerField(default=0, verbose_name='데이터 생성 횟수')

    class Meta:
        db_table = 'motion_ai'
        verbose_name = '모션 AI'
        verbose_name_plural = '모션 AI'


class TempData1(models.Model):
    motion_name1 = models.CharField(max_length=256,
                                    verbose_name='모션 이름1')
    motion_name2 = models.CharField(max_length=256,
                                    verbose_name='모션 이름2')
    motion_name3 = models.CharField(max_length=256,
                                    verbose_name='모션 이름3')
    motion_name4 = models.CharField(max_length=256,
                                    verbose_name='모션 이름4')
    motion_name5 = models.CharField(max_length=256,
                                    verbose_name='모션 이름5')
    motion_name6 = models.CharField(max_length=256,
                                    verbose_name='모션 이름6')
    motion_writer = models.ForeignKey('fcuser.Fcuser', on_delete=models.CASCADE,
                                      verbose_name='작성자')

    class Meta:
        db_table = 'TempDate1'
        verbose_name = '임시 저장'
        verbose_name_plural = '임시 저장'


class TempData2(models.Model):
    session_num = models.IntegerField(default = 0, verbose_name='로그인 실패 횟수')
    motion_writer = models.ForeignKey('fcuser.Fcuser', on_delete=models.CASCADE,
                                      verbose_name='작성자')

    class Meta:
        db_table = 'TempDate2'
        verbose_name = '가상 데이터 임시 저장'
        verbose_name_plural = '가상 데이터 임시 저장'