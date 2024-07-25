# 필요한 라이브러리 import
from audiocraft.models import MusicGen
from audiocraft.models import MultiBandDiffusion
import math
import torchaudio
import torch
from audiocraft.data.audio import audio_write
import boto3

# MultiBandDiffusion 디코더 사용 여부 설정
USE_DIFFUSION_DECODER = False

# MusicGen 모델 초기화 (large 모델 사용)
model = MusicGen.get_pretrained('facebook/musicgen-large')
if USE_DIFFUSION_DECODER:
    mbd = MultiBandDiffusion.get_mbd_musicgen()

# 오디오 생성 파라미터 설정
model.set_generation_params(
    use_sampling=True,  # 샘플링 기반 생성 방식 사용
    top_k=250,  # 상위 250개 토큰만 고려
    duration=30  # 생성할 오디오 길이 30초
)

# AWS S3 클라이언트 초기화 및 버킷 이름 설정
s3 = boto3.client('s3')
bucket_name = 'chatbot-flask'  # S3 버킷 이름

# 440Hz 비프 소리 생성 함수
def get_bip_bip(bip_duration=0.125, frequency=440,
                duration=0.5, sample_rate=32000, device="cuda"):
    """440Hz 비프 소리를 생성하는 함수"""
    t = torch.arange(
        int(duration * sample_rate), device="cuda", dtype=torch.float) / sample_rate
    wav = torch.cos(2 * math.pi * 440 * t)[None]
    tp = (t % (2 * bip_duration)) / (2 * bip_duration)
    envelope = (tp >= 0.5).float()
    return wav * envelope

'''
joy 
hope 
neutrality
sadness 
anger
anxiety 
tiredness
regret
'''

res = model.generate_continuation(
    get_bip_bip(0.125).expand(8, -1, -1),  # prompt의 길이는 3
    32000, [
        'Joyful upbeat music with cheerful melodies and harmonies',
        'Inspirational music with uplifting melodies and motivational lyrics' ,
        'Calm and serene ambient music for relaxation and meditation' ,
        'Melancholic and emotional ballads with heartfelt lyrics' ,
        'Intense and aggressive rock music with heavy riffs and powerful drums' ,
        'Atmospheric and unsettling ambient music with tense and suspenseful vibes' ,
        'Soothing and gentle lullabies with calming melodies and soft harmonies' ,
        'Melancholic and introspective music with somber melodies and reflective lyrics', 
    
    ],
    progress=True)

# 생성된 오디오 파일 저장 및 S3 업로드
for idx, one_wav in enumerate(res):
    # 오디오 파일 저장 ({idx}.wav)
    audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    # S3에 WAV 파일 업로드
    file_name = f'{idx}.wav'
    s3.upload_file(file_name, bucket_name, file_name)
    print(f'{file_name} 파일이 S3 버킷에 업로드되었습니다.')
