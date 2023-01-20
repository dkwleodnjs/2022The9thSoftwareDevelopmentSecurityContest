from django.shortcuts import render, redirect
from fcuser.models import Fcuser
from .models import MotionAI, TempData1, TempData2
from .forms import MotionAIForm
import cv2
import mediapipe as mp
import numpy as np
import time
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from hashlib import sha256
from django.views.decorators import gzip
from django.http import StreamingHttpResponse
from time import sleep
from e_drone.drone import *
from e_drone.protocol import *
import requests
import json
import os


# Create your views here.
def authentication_model_gesture(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    if request.method == 'POST':
        return render(request, 'authentication.html')
    else:
        exist = 0
        request.session['exist'] = exist
    return render(request, 'authentication.html')


def learned_model_gesture(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    if request.method == 'POST':
        return render(request, 'learnedmodel.html')
    else:
        exist = 0
        request.session['exist'] = exist
    return render(request, 'learnedmodel.html')


def gesture_register(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    if request.method == 'POST':
        user_id = request.session.get('user')
        fcuser = Fcuser.objects.get(pk=user_id)
        motion_name1 = request.POST.get('motion_name1', None)
        motion_name2 = request.POST.get('motion_name2', None)
        motion_name3 = request.POST.get('motion_name3', None)
        motion_name4 = request.POST.get('motion_name4', None)
        motion_name5 = request.POST.get('motion_name5', None)
        motion_name6 = request.POST.get('motion_name6', None)

        tempdata = TempData1()
        res_data = {}
        if motion_name1 != '' and motion_name2 != '' and motion_name3 != '' and motion_name4 != '' and motion_name5 != '' and motion_name6 != '':
            tempdata.motion_name1 = str(motion_name1)
            tempdata.motion_name2 = str(motion_name2)
            tempdata.motion_name3 = str(motion_name3)
            tempdata.motion_name4 = str(motion_name4)
            tempdata.motion_name5 = str(motion_name5)
            tempdata.motion_name6 = str(motion_name6)
            tempdata.motion_writer = fcuser
            tempdata.save()
            exist = 1
            request.session['exist'] = exist

        elif motion_name1 != '' and motion_name2 != '' and motion_name3 != '' and motion_name4 != '' and motion_name5 != '' and motion_name6 == '':
            tempdata.motion_name1 = str(motion_name1)
            tempdata.motion_name2 = str(motion_name2)
            tempdata.motion_name3 = str(motion_name3)
            tempdata.motion_name4 = str(motion_name4)
            tempdata.motion_name5 = str(motion_name5)
            tempdata.motion_name6 = str('')
            tempdata.motion_writer = fcuser
            tempdata.save()
            exist = 1
            request.session['exist'] = exist

        elif motion_name1 != '' and motion_name2 != '' and motion_name3 != '' and motion_name4 != '' and motion_name5 == '' and motion_name6 == '':
            tempdata.motion_name1 = str(motion_name1)
            tempdata.motion_name2 = str(motion_name2)
            tempdata.motion_name3 = str(motion_name3)
            tempdata.motion_name4 = str(motion_name4)
            tempdata.motion_name5 = str('')
            tempdata.motion_name6 = str('')
            tempdata.motion_writer = fcuser
            tempdata.save()
            exist = 1
            request.session['exist'] = exist

        elif motion_name1 != '' and motion_name2 != '' and motion_name3 != '' and motion_name4 == '' and motion_name5 == '' and motion_name6 == '':
            tempdata.motion_name1 = str(motion_name1)
            tempdata.motion_name2 = str(motion_name2)
            tempdata.motion_name3 = str(motion_name3)
            tempdata.motion_name4 = str('')
            tempdata.motion_name5 = str('')
            tempdata.motion_name6 = str('')
            tempdata.motion_writer = fcuser
            tempdata.save()
            exist = 1
            request.session['exist'] = exist

        elif motion_name1 != '' and motion_name2 != '' and motion_name3 == '' and motion_name4 == '' and motion_name5 == '' and motion_name6 == '':
            tempdata.motion_name1 = str(motion_name1)
            tempdata.motion_name2 = str(motion_name2)
            tempdata.motion_name3 = str('')
            tempdata.motion_name4 = str('')
            tempdata.motion_name5 = str('')
            tempdata.motion_name6 = str('')
            tempdata.motion_writer = fcuser
            tempdata.save()
            exist = 1
            request.session['exist'] = exist

        elif motion_name1 != '' and motion_name2 == '' and motion_name3 == '' and motion_name4 == '' and motion_name5 == '' and motion_name6 == '':
            exist = 0
            request.session['exist'] = exist
            res_data['error'] = '최소 2개 이상의 데이터를 입력해주세요.'

        elif motion_name1 == '' and motion_name2 == '' and motion_name3 == '' and motion_name4 == '' and motion_name5 == '' and motion_name6 == '':
            exist = 0
            request.session['exist'] = exist
            res_data['error'] = '최소 2개 이상의 데이터를 입력해주세요.'
        else:
            exist = 0
            request.session['exist'] = exist
            res_data['error'] = '순서대로 입력해주세요.'

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
                "text": f'{fcuser} 모델을 생성 했습니다.',
                "link": {
                    "web_url": "www.naver.com"
                }
            })
        }

        response = requests.post(url, headers=headers, data=data)
        response.status_code
        
        return render(request, 'airegister.html', res_data)
    else:
        exist = 0
        request.session['exist'] = exist
        form = MotionAIForm()
        
    return render(request, 'airegister.html', {'form': form})


def make_hash(data: str, prev_hash: bytes) -> bytes:
    return sha256(data.encode() + prev_hash).digest()


def blockchain_make(action, motion_writer):
    last_blockchain = MotionAI.objects.last()

    if last_blockchain == None:
        data = action
        prev_hash = ''
        prev_hash = bytes(prev_hash, 'utf-8')
        current_hash = make_hash(data, prev_hash)
        motionAI = MotionAI()
        motionAI.motion_name = action
        motionAI.prev_hash = current_hash
        motionAI.current_hash = current_hash
        motionAI.make_data_count = 1
        motionAI.motion_writer = motion_writer
        motionAI.save()

        return True

    elif last_blockchain != None:
        data = action
        prev_hash = last_blockchain.current_hash
        current_hash = make_hash(data, prev_hash)

        if verify_blockchain(last_blockchain.make_data_count):
            motionAI = MotionAI()
            motionAI.motion_name = action
            motionAI.prev_hash = prev_hash
            motionAI.current_hash = current_hash
            make_data_count = last_blockchain.make_data_count
            motionAI.make_data_count = make_data_count + 1
            motionAI.motion_writer = motion_writer
            motionAI.save()

            return True

    return False


def verify_blockchain(make_data_count):
    if (make_data_count-1) <= 1:
        return True

    for i in range(2, make_data_count):
        blockchain_posite1 = MotionAI.objects.get(id=make_data_count)
        last_data1 = blockchain_posite1.motion_name
        last_prev_hash1 = blockchain_posite1.prev_hash
        last_current_hash1 = blockchain_posite1.current_hash

        blockchain_posite2 = MotionAI.objects.get(id=make_data_count-1)
        last_data2 = blockchain_posite2.motion_name
        last_prev_hash2 = blockchain_posite2.prev_hash
        last_current_hash2 = blockchain_posite2.current_hash

        if last_prev_hash1 != last_current_hash2:
            return False

        if last_current_hash2 != (temp := make_hash(last_data2, last_prev_hash2)):
            return False

        if last_current_hash1 != (temp := make_hash(last_data1, last_prev_hash1)):
            return False

        return True


# 초기 영상
def InitvideoMotionDraw():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, img = cap.read()

        img = cv2.putText(cv2.flip(img, 1), f'Connect to Success!', org=(
            10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)

        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        _, jpeg = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@gzip.gzip_page
def Initdetectme(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')
    return StreamingHttpResponse(InitvideoMotionDraw(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


# 학습된 영상
def learnedvideoMotionDraw(fcuser):
    actions = []

    tmp = TempData1.objects.filter(motion_writer=fcuser).last()
    if tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 != '' and tmp.motion_name6 != '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))
        actions.append(str(tmp.motion_name5))
        actions.append(str(tmp.motion_name6))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 != '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))
        actions.append(str(tmp.motion_name5))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 == '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 == '' and tmp.motion_name4 == '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))

    return aiLearned(actions, fcuser)


def aiLearned(actions, fcuser):

    seq_length = 40
    model = load_model(f'static/model{fcuser}.h5')

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    seq = []
    action_seq = []
    cap = cv2.VideoCapture(0)
    last_action = None

    while cap.isOpened():
        # 영상
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                # 손 인식
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10,
                            11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                angle = np.degrees(angle)

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue
                input_data = np.expand_dims(
                    np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = ''
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                    if last_action != this_action:
                        tempdata2 = TempData2.objects.filter(
                            motion_writer=fcuser).last()
                        if this_action == 'go':
                            if tempdata2 != None:
                                if tempdata2.session_num == 1:
                                    drone = Drone()
                                    drone.open("COM9")
                                    print("TakeOff")
                                    drone.sendTakeOff()
                                    sleep(0.01)

                                    print("Hovering")
                                    drone.sendControlWhile(0, 0, 0, 0, 2000)
                                    sleep(2)

                                    print("Go")
                                    drone.sendControlWhile(0, 50, 0, 0, 1000)
                                    sleep(0.01)

                                    print("Hovering")
                                    drone.sendControlWhile(0, 0, 0, 0, 1000)
                                    sleep(0.01)

                                    print("Landing")
                                    drone.sendLanding()
                                    sleep(0.01)
                                    drone.sendLanding()
                                    sleep(0.01)

                                    drone.close()

                        elif this_action == 'roll':
                            if tempdata2.session_num == 1:
                                drone = Drone()
                                drone.open("COM9")
                                print("TakeOff")
                                drone.sendTakeOff()
                                sleep(0.01)

                                print("Hovering")
                                drone.sendControlWhile(0, 0, 0, 0, 2000)
                                sleep(2)

                                print("Flip")
                                drone.sendFlightEvent(FlightEvent.FlipFront)
                                sleep(1)

                                print("Hovering")
                                drone.sendControlWhile(0, 0, 0, 0, 1000)
                                sleep(0.01)

                                print("Landing")
                                drone.sendLanding()
                                sleep(0.01)
                                drone.sendLanding()
                                sleep(0.01)

                                drone.close()

                        elif this_action == 'back':
                            if tempdata2.session_num == 1:
                                drone = Drone()
                                drone.open("COM9")
                                print("TakeOff")
                                drone.sendTakeOff()
                                sleep(0.01)

                                print("Hovering")
                                drone.sendControlWhile(0, 0, 0, 0, 2000)
                                sleep(2)

                                print("Back")
                                drone.sendControlWhile(0, -70, 0, 0, 1000)
                                sleep(0.01)

                                print("Hovering")
                                drone.sendControlWhile(0, 0, 0, 0, 1000)
                                sleep(0.01)

                                print("Landing")
                                drone.sendLanding()
                                sleep(0.01)
                                drone.sendLanding()
                                sleep(0.01)

                                drone.close()
                    last_action = this_action
                if this_action == "me":
                    this_action = ""
                else:
                    this_action = ""

                img = cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(
                    res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, jpeg = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@gzip.gzip_page
def learneddetectme(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    user_id = request.session.get('user')
    fcuser = Fcuser.objects.get(pk=user_id)
    return StreamingHttpResponse(learnedvideoMotionDraw(fcuser),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


# 학습하는 영상
def learningvideoMotionDraw():
    tmp = TempData1.objects.last()
    fcuser = tmp.motion_writer

    actions = []

    if tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 != '' and tmp.motion_name6 != '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))
        actions.append(str(tmp.motion_name5))
        actions.append(str(tmp.motion_name6))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 != '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))
        actions.append(str(tmp.motion_name5))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 == '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 == '' and tmp.motion_name4 == '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))

    seq_length = 40
    secs_for_action = 10

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)
    created_time = int(time.time())
    temp = []

    while cap.isOpened():
        for idx, action in enumerate(actions):
            data = []

            ret, img = cap.read()
            img = cv2.flip(img, 1)

            cv2.waitKey(1500)

            start_time = time.time()

            while time.time() - start_time < secs_for_action:
                ret, img = cap.read()

                img = cv2.flip(img, 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                result = hands.process(img)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                if result.multi_hand_landmarks is not None:
                    for res in result.multi_hand_landmarks:
                        joint = np.zeros((21, 4))
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9,
                                    10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                    11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                        v = v2 - v1
                        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]
                        angle = np.arccos(np.einsum('nt,nt->n',
                                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                        12, 13, 14, 16, 17, 18], :],
                                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                        angle = np.degrees(angle)

                        angle_label = np.array([angle], dtype=np.float32)
                        angle_label = np.append(angle_label, idx)

                        d = np.concatenate([joint.flatten(), angle_label])

                        data.append(d)

                        mp_drawing.draw_landmarks(
                            img, res, mp_hands.HAND_CONNECTIONS)

                _, jpeg = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            data = np.array(data)
            full_seq_data = []
            for seq in range(len(data) - seq_length):
                full_seq_data.append(data[seq:seq + seq_length])

            full_seq_data = np.array(full_seq_data)
            tmp = f'seq_{action}_{created_time}'
            temp.append(tmp)
            np.save(os.path.join('static/', tmp), full_seq_data)
        cap.release()

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    judge = False
    if len(temp) == 6:
        data = np.concatenate([
            np.load(os.path.join('static/', f'{temp[0]}.npy')),
            np.load(os.path.join('static/', f'{temp[1]}.npy')),
            np.load(os.path.join('static/', f'{temp[2]}.npy')),
            np.load(os.path.join('static/', f'{temp[3]}.npy')),
            np.load(os.path.join('static/', f'{temp[4]}.npy')),
            np.load(os.path.join('static/', f'{temp[5]}.npy')),
        ], axis=0)
        tmp = TempData1.objects.last()
        judge = blockchain_make(tmp.motion_name6, tmp.motion_writer)
    elif len(temp) == 5:
        data = np.concatenate([
            np.load(os.path.join('static/', f'{temp[0]}.npy')),
            np.load(os.path.join('static/', f'{temp[1]}.npy')),
            np.load(os.path.join('static/', f'{temp[2]}.npy')),
            np.load(os.path.join('static/', f'{temp[3]}.npy')),
            np.load(os.path.join('static/', f'{temp[4]}.npy')),
        ], axis=0)
        tmp = TempData1.objects.last()
        judge = blockchain_make(tmp.motion_name5, tmp.motion_writer)
    elif len(temp) == 4:
        data = np.concatenate([
            np.load(os.path.join('static/', f'{temp[0]}.npy')),
            np.load(os.path.join('static/', f'{temp[1]}.npy')),
            np.load(os.path.join('static/', f'{temp[2]}.npy')),
            np.load(os.path.join('static/', f'{temp[3]}.npy')),
        ], axis=0)
        tmp = TempData1.objects.last()
        judge = blockchain_make(tmp.motion_name4, tmp.motion_writer)
    elif len(temp) == 3:
        data = np.concatenate([
            np.load(os.path.join('static/', f'{temp[0]}.npy')),
            np.load(os.path.join('static/', f'{temp[1]}.npy')),
            np.load(os.path.join('static/', f'{temp[2]}.npy')),
        ], axis=0)
        tmp = TempData1.objects.last()
        judge = blockchain_make(tmp.motion_name3, tmp.motion_writer)
    elif len(temp) == 2:
        data = np.concatenate([
            np.load(os.path.join('static/', f'{temp[0]}.npy')),
            np.load(os.path.join('static/', f'{temp[1]}.npy')),
        ], axis=0)
        tmp = TempData1.objects.last()
        judge = blockchain_make(tmp.motion_name2, tmp.motion_writer)

    if judge == True:
        x_data = data[:, :, :-1]
        labels = data[:, 0, -1]

        y_data = to_categorical(labels, num_classes=len(actions))
        x_data = x_data.astype(np.float32)
        y_data = y_data.astype(np.float32)

        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data, test_size=0.1, random_state=2021)

        model = Sequential([
            LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
            Dense(32, activation='relu'),
            Dense(len(actions), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['acc'])

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=150,
            callbacks=[
                ModelCheckpoint(f'static/model{fcuser}.h5', monitor='val_acc',
                                verbose=1, save_best_only=True, mode='auto'),
                ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                                  patience=50, verbose=1, mode='auto')
            ]
        )


@gzip.gzip_page
def learningdetectme(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    return StreamingHttpResponse(learningvideoMotionDraw(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')


def authenticationvideoMotionDraw(fcuser):
    actions = []

    tmp = TempData1.objects.filter(motion_writer=fcuser).last()
    if tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 != '' and tmp.motion_name6 != '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))
        actions.append(str(tmp.motion_name5))
        actions.append(str(tmp.motion_name6))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 != '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))
        actions.append(str(tmp.motion_name5))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 != '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))
        actions.append(str(tmp.motion_name4))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 != '' and tmp.motion_name4 == '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))
        actions.append(str(tmp.motion_name3))

    elif tmp.motion_name1 != '' and tmp.motion_name2 != '' and tmp.motion_name3 == '' and tmp.motion_name4 == '' and tmp.motion_name5 == '' and tmp.motion_name6 == '':
        actions.append(str(tmp.motion_name1))
        actions.append(str(tmp.motion_name2))

    return authentication(actions, fcuser)


def authentication(actions, fcuser):

    seq_length = 40
    model = load_model(f'static/model{fcuser}.h5')

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    seq = []
    action_seq = []
    cap = cv2.VideoCapture(0)
    last_action = None

    while cap.isOpened():
        # 영상
        ret, img = cap.read()
        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                # 손 인식
                joint = np.zeros((21, 4))
                for j, lm in enumerate(res.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10,
                            11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                            12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                v = v2 - v1
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10,
                                                12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))

                angle = np.degrees(angle)

                d = np.concatenate([joint.flatten(), angle])

                seq.append(d)

                mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue
                input_data = np.expand_dims(
                    np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.9:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 3:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3]:
                    this_action = action

                    if last_action != this_action:
                        tempdata2 = TempData2.objects.filter(
                            motion_writer=fcuser).last()
                        if this_action == 'me':
                            if tempdata2 != None:
                                tempdata2.session_num = 1
                                tempdata2.motion_writer = fcuser
                                tempdata2.save()

                            elif tempdata2 == None:
                                tempdata2 = TempData2()
                                tempdata2.session_num = 1
                                tempdata2.motion_writer = fcuser
                                tempdata2.save()
                    last_action = this_action
                this_action = ''

                img = cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(
                    res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        _, jpeg = cv2.imencode('.jpg', img)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')


@gzip.gzip_page
def authenticationdetectme(request):
    if not request.session.get('user'):
        return redirect('/fcuser/login/')

    user_id = request.session.get('user')
    fcuser = Fcuser.objects.get(pk=user_id)
    return StreamingHttpResponse(authenticationvideoMotionDraw(fcuser),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
