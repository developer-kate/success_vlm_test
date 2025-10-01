import os
import json
import time
import platform
import cv2
import mediapipe as mp
import numpy as np
import base64
import requests
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
load_dotenv()

print("작업자 행동 분석 시스템 v9.0 - VLM 기반 분석")
print("=" * 70)

# API 설정
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
#GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


class WorkerAnalysisSystem:
    """통합 작업자 분석 시스템"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        
        # 비디오 FPS 정보 저장
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30.0
        cap.release()
        print(f"비디오 FPS: {self.fps}")
        
        self.robot_actions = {
            "grip_open": {"name": "그리퍼 열기", "description": "그리퍼를 열어 부품을 놓을 준비", "robot_command": "grip.set(100)"},
            "grip_close": {"name": "그리퍼 닫기", "description": "그리퍼를 닫아 부품을 잡음", "robot_command": "grip.set(0)"},
            "move_joint": {"name": "로봇 관절 이동", "description": "특정 관절 값으로 이동", "robot_command": "robot.MoveJ(...)"},
            "move_linear": {"name": "직선 이동", "description": "특정 좌표로 직선 이동", "robot_command": "robot.MoveL(...)"}
        }
        # GEMINI_API_URL과 GEMINI_API_KEY를 클래스 인스턴스 변수로 저장
        self.GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        
        self.analysis_types = ['full_analysis', 'robot_analysis']
        self.log_file = f'logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}_analysis.log'
        self.results_path = f'results/{os.path.basename(self.video_path).split(".")[0]}_analysis.json'
        
        # 새로운 이벤트 타입 정의
        self.event_types = [
            "assemble_system", "take_subsystem", "put_down_subsystem", "consult_sheets", "turn_sheets",
            "take_screwdriver", "put_down_screwdriver", "picking_in_front", "picking_left",
            "take_measuring_rod", "put_down_measuring_rod", "meta_action"
        ]
        
    def _log(self, message: str, level: str = 'info'):
        """분석 로그 기록"""
        if level == 'info':
            logging.info(message)
        elif level == 'error':
            logging.error(message)

    def _extract_skeleton_data(self, video_path: str) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
        """비디오에서 스켈레톤 데이터와 프레임을 추출"""
        self._log(f"비디오 '{os.path.basename(video_path)}'에서 스켈레톤 데이터 추출 중...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self._log(f"비디오 파일을 열 수 없습니다: {video_path}", 'error')
            return [], []

        frames = []
        skeleton_data = []
        frame_idx = 0

        with self.mp_pose.Pose() as pose:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frames.append(frame)
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                landmarks = {}
                if results.pose_landmarks:
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        landmarks[id] = {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
                
                skeleton_data.append(landmarks)
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    self._log(f"프레임 {frame_idx} 처리 완료...")

        cap.release()
        self._log(f"{len(frames)}개 프레임에서 스켈레톤 데이터 추출 완료.")
        return skeleton_data, frames

    def _detect_movement_change_points(self, skeleton_data: List[Dict[str, Any]]) -> List[int]:
        """손목 움직임 변화점을 감지하여 세그먼트 경계 식별"""
        self._log("손목 움직임 변화점 감지 중...")
        
        change_points = [0]
        l_wrist_idx = self.mp_pose.PoseLandmark.LEFT_WRIST.value
        r_wrist_idx = self.mp_pose.PoseLandmark.RIGHT_WRIST.value

        for i in range(1, len(skeleton_data) - 1):
            # 이전, 현재, 다음 프레임 모두에서 양쪽 손목 데이터가 있는지 확인
            if (l_wrist_idx in skeleton_data[i-1] and r_wrist_idx in skeleton_data[i-1] and
                l_wrist_idx in skeleton_data[i] and r_wrist_idx in skeleton_data[i] and
                l_wrist_idx in skeleton_data[i+1] and r_wrist_idx in skeleton_data[i+1]):

                prev_l_wrist = np.array([skeleton_data[i-1][l_wrist_idx]['x'], skeleton_data[i-1][l_wrist_idx]['y']])
                curr_l_wrist = np.array([skeleton_data[i][l_wrist_idx]['x'], skeleton_data[i][l_wrist_idx]['y']])
                next_l_wrist = np.array([skeleton_data[i+1][l_wrist_idx]['x'], skeleton_data[i+1][l_wrist_idx]['y']])

                prev_r_wrist = np.array([skeleton_data[i-1][r_wrist_idx]['x'], skeleton_data[i-1][r_wrist_idx]['y']])
                curr_r_wrist = np.array([skeleton_data[i][r_wrist_idx]['x'], skeleton_data[i][r_wrist_idx]['y']])
                next_r_wrist = np.array([skeleton_data[i+1][r_wrist_idx]['x'], skeleton_data[i+1][r_wrist_idx]['y']])

                l_vel_curr = np.linalg.norm(curr_l_wrist - prev_l_wrist)
                l_vel_next = np.linalg.norm(next_l_wrist - curr_l_wrist)

                r_vel_curr = np.linalg.norm(curr_r_wrist - prev_r_wrist)
                r_vel_next = np.linalg.norm(next_r_wrist - curr_r_wrist)
                
                if (l_vel_curr > 0.01 and l_vel_next < 0.005) or \
                   (l_vel_curr < 0.005 and l_vel_next > 0.01) or \
                   (r_vel_curr > 0.01 and r_vel_next < 0.005) or \
                   (r_vel_curr < 0.005 and r_vel_next > 0.01):
                    if len(change_points) == 0 or i > change_points[-1] + 10:
                        change_points.append(i)

        change_points.append(len(skeleton_data) - 1)
        self._log(f"감지된 변화점: {change_points}")
        return change_points

    def _create_events_from_change_points(self, change_points: List[int], frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """변화점을 기반으로 이벤트 세그먼트 생성"""
        self._log("이벤트 세그먼트 생성 중...")
        events = []
        for i in range(len(change_points) - 1):
            start_frame_idx = change_points[i]
            end_frame_idx = change_points[i+1]
            
            # 이벤트 세그먼트가 너무 짧은 경우 건너뛰기
            if end_frame_idx - start_frame_idx < 10:
                continue

            # 프레임을 시간으로 변환
            start_time = start_frame_idx / self.fps
            end_time = end_frame_idx / self.fps
            duration = end_time - start_time

            event = {
                'start_frame': start_frame_idx,
                'end_frame': end_frame_idx,
                'start_time': start_time,
                'end_time': end_time,
                'duration': duration,
                'type': 'unclassified',
                'confidence': 0.0,
                'description': ''
            }
            events.append(event)
            
        self._log(f"{len(events)}개의 이벤트 세그먼트 생성 완료.")
        return events

    def _encode_image_to_base64(self, image: np.ndarray) -> str:
        """OpenCV 이미지를 Base64 문자열로 인코딩"""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    
    def _enhance_events_with_gemini(self, events: List[Dict[str, Any]], frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Gemini API를 사용하여 이벤트 분류 및 설명 보강
        이 메서드가 이제 주된 분류 역할을 수행
        """
        self._log("Gemini VLM을 사용하여 이벤트 분석 및 분류 중...")
        
        classified_events = []
        for i, event in enumerate(events):
            start_frame_idx = event['start_frame']
            end_frame_idx = event['end_frame']
            
            # 프레임 수가 부족하면 중간 프레임만 사용
            if end_frame_idx - start_frame_idx <= 1:
                frame_indices_to_analyze = [start_frame_idx]
            else:
                # 시작, 중간, 끝 프레임 인덱스 계산
                mid_frame_idx = start_frame_idx + (end_frame_idx - start_frame_idx) // 2
                frame_indices_to_analyze = [start_frame_idx, mid_frame_idx, end_frame_idx]

            prompt_parts = [
                {"text": "비디오의 연속된 프레임들입니다. 이 프레임들을 보고 작업자가 수행하는 동작을 가장 적절하게 설명하고, 다음 12가지 동작 중 하나로 분류해주세요:"},
                {"text": ", ".join(self.event_types)},
                {"text": f"\n\n다음 JSON 형식으로만 응답하세요: {{\"type\": \"동작_타입\"}}"},
                {"text": "동작에 대한 설명은 'pick up a part from the left', 'put down the screw driver', 등과 같이 구체적으로 작성해 주세요."},
            ]

            # 프레임 이미지를 base64로 인코딩하여 prompt_parts에 추가
            for frame_idx in frame_indices_to_analyze:
                image_b64 = self._encode_image_to_base64(frames[frame_idx])
                prompt_parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_b64}})

            request_body = {
                "contents": [{"parts": prompt_parts}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 2048
                }
            }
            
            headers = {"Content-Type": "application/json"}

            # 지수 백오프 로직 추가
            max_retries = 3
            wait_time = 5
            
            for attempt in range(max_retries):
                try:
                    self._log(f"이벤트 {i+1}/{len(events)} 분석 중... (프레임 {start_frame_idx}~{end_frame_idx}, 시간 {event['start_time']:.2f}~{event['end_time']:.2f}s)")
                    response = requests.post(f"{self.GEMINI_API_URL}?key={self.GEMINI_API_KEY}", headers=headers, json=request_body, timeout=30)
                    response.raise_for_status()
                    
                    response_data = response.json()
                    
                    # 새로운 안전 장치: 'candidates' 키가 있는지 확인
                    if 'candidates' in response_data and len(response_data['candidates']) > 0:
                        text_response = response_data['candidates'][0]['content']['parts'][0]['text']
                        
                        json_start = text_response.find('{')
                        json_end = text_response.rfind('}') + 1
                        if json_start != -1 and json_end != -1:
                            json_str = text_response[json_start:json_end]
                            gemini_result = json.loads(json_str)
                            
                            if 'type' in gemini_result and gemini_result['type'] in self.event_types:
                                event['type'] = gemini_result['type']
                                event['description'] = ''
                                event['confidence'] = 1.0
                            else:
                                event['type'] = 'unknown'
                                event['description'] = f"Gemini API가 알 수 없는 타입 응답: {gemini_result.get('type')}"
                        else:
                            event['type'] = 'unknown'
                            event['description'] = f"Gemini API 응답에서 JSON 파싱 오류: {text_response}"
                    else:
                        # API가 오류 메시지 등을 반환한 경우
                        error_message = response_data.get('error', {}).get('message', '알 수 없는 API 응답')
                        self._log(f"Gemini API가 유효한 응답을 반환하지 않았습니다: {error_message}", 'error')
                        event['type'] = 'error'
                        event['description'] = f"API 응답 오류: {error_message}"
                        
                    break  # 성공 또는 처리 가능한 오류 시 루프 탈출
                
                except requests.exceptions.HTTPError as err:
                    if err.response.status_code == 429 and attempt < max_retries - 1:
                        self._log(f"API 호출 횟수 초과 (429). {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})", 'error')
                        time.sleep(wait_time)
                        wait_time *= 2
                    else:
                        self._log(f"Gemini API 호출 중 오류 발생: {err}", 'error')
                        event['type'] = 'error'
                        event['description'] = f"API 호출 오류: {err}"
                        break
                except Exception as e:
                    self._log(f"Gemini API 호출 중 예기치 않은 오류 발생: {e}", 'error')
                    event['type'] = 'error'
                    event['description'] = f"API 호출 오류: {e}"
                    break
            
            classified_events.append(event)
        
        self._log("Gemini 기반 이벤트 분류 완료.")
        return classified_events
    
    def analyze_video(self) -> List[Dict[str, Any]]:
        """메인 비디오 분석 로직"""
        try:
            skeleton_data, frames = self._extract_skeleton_data(self.video_path)
            if not skeleton_data:
                return []
            
            change_points = self._detect_movement_change_points(skeleton_data)
            events = self._create_events_from_change_points(change_points, frames)
            
            # Gemini를 사용하여 이벤트 1차 분류 및 보강
            enhanced_events = self._enhance_events_with_gemini(events, frames)
            
            return enhanced_events
            
        except Exception as e:
            self._log(f"비디오 분석 중 치명적인 오류 발생: {e}", 'error')
            return []

    def select_robot_actions(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """분석된 이벤트에서 로봇 동작을 선택"""
        self._log("로봇 동작 이벤트 선택 중...")
        robot_events = []
        for event in events:
            # 예: picking, put_down, take 동작을 로봇 동작과 연결
            if 'picking' in event['type'] or 'put_down' in event['type'] or 'take_' in event['type']:
                # 여기서 VLM/LLM을 활용하여 더 정교한 로봇 동작 연결 로직을 구현할 수 있음
                # 예: Gemini에게 '이 동작 다음에 그리퍼를 닫아야 하는가?' 질문
                # 현재는 단순히 동작 타입에 따라 선택
                robot_events.append(event)
        
        self._log(f"총 {len(robot_events)}개의 로봇 동작 이벤트 선택 완료.")
        return robot_events

    def generate_robot_script(self, analysis_results: Dict[str, Any]) -> str:
        """분석 결과를 기반으로 RoboDK 파이썬 스크립트 생성"""
        self._log("로봇 제어 스크립트 생성 중...")
        
        # 스크립트 파일명 설정
        script_filename = f"robot_scripts/{os.path.basename(self.video_path).split('.')[0]}_robot_script_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
        
        script_content = f"""
# Gemini VLM 분석 결과를 기반으로 RoboDK 로봇을 제어합니다.
# 생성 날짜: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import sys
import time
from robolink import *
from robodk import *

# 로봇 연결
RDK = Robolink()
robot = RDK.Item('UR10e', ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("로봇을 찾을 수 없습니다. 'UR10e' 로봇이 RoboDK 스테이션에 있는지 확인하세요.")

# 로봇 시작점 설정 (필요시 수정)
home_pose = robot.Pose() # 현재 위치를 시작 위치로 설정
safe_pose = KUKA_2(1,1,1,1,1,1) # 예시 안전 위치, 실제 로봇에 맞게 수정 필요

# 메인 실행 시퀀스
def execute_analyzed_sequence():
    print("로봇 제어 시퀀스 시작")
    robot.setSpeed(100) # mm/s
    robot.setJoints(robot.PoseJ()) # 현재 위치에서 시작

    try:
        # 분석된 이벤트에 따라 로봇 동작 실행
        print("분석된 이벤트에 따라 동작 실행 중...")
        
        # 이벤트 분석 결과
        analyzed_events = {json.dumps(analysis_results['robot_selected_events'], indent=4)}
        
        # TODO: 실제 로봇 제어 로직을 추가해야 합니다.
        # 아래는 예시 코드입니다.
        for event in analyzed_events:
            print(f"-> 동작 감지: {{event['type']}} (프레임 {{event['start_frame']}}~{{event['end_frame']}})")
            print(f"설명: {{event['description']}}")
            # 예: picking_in_front 동작에 대한 로봇 제어 명령
            if event['type'] == 'picking_in_front':
                # 가상의 앞쪽 부품 위치로 이동
                target_pose = KUKA_2(500, -200, 200, 180, 0, 180) 
                robot.MoveL(target_pose)
                # 그리퍼 닫기
                # grip.set(0) # 실제 RoboDK 그리퍼 제어 명령으로 교체
                
            elif event['type'] == 'picking_left':
                # 가상의 왼쪽 부품 위치로 이동
                target_pose = KUKA_2(300, 400, 200, 180, 0, 180)
                robot.MoveL(target_pose)
                # 그리퍼 닫기
                # grip.set(0)
            
            # 기타 동작에 대한 로봇 제어 로직 추가
            time.sleep(1.0) # 실제 동작 시간 고려
            
        # 작업 완료 후 홈 포지션으로 복귀
        print("작업 완료 - 홈 포지션으로 복귀")
        robot.MoveJ(home_pose)
        
    except Exception as e:
        print(f"로봇 제어 중 오류 발생: {{e}}")
        # 안전 위치로 이동
        robot.MoveJ(safe_pose)

# 메인 실행 함수
if __name__ == "__main__":
    execute_analyzed_sequence()

"""
        
        # 스크립트 파일 저장
        try:
            os.makedirs("robot_scripts", exist_ok=True)
            with open(script_filename, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            self._log(f"로봇 스크립트가 성공적으로 생성되었습니다: {script_filename}")
            return script_filename
        except Exception as e:
            self._log(f"스크립트 파일 저장 중 오류 발생: {e}", 'error')
            return ""      
