import os
os.environ['MPLCONFIGDIR']='/tmp/greeny-mpl-cache'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams['svg.fonttype']='path'
fig, ax=plt.subplots(figsize=(17,11.5),dpi=150)
fig.patch.set_facecolor('#f8fafc'); ax.set_facecolor('#f8fafc')
ax.set_xlim(0,17);ax.set_ylim(0,11.5);ax.axis('off')
def txt(x,y,s,size=11,color='#172b4d',ha='left',weight='normal'):
    ax.text(x,y,s,fontproperties=font,fontsize=size,color=color,ha=ha,va='center',weight=weight)
def box(x,y,w,h,title,body,fill='#ffffff',edge='#b9c8db'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.12',facecolor=fill,edgecolor=edge,lw=1.2))
    txt(x+w/2,y+h*.72,title,12,ha='center',weight='bold')
    txt(x+w/2,y+h*.32,body,10,ha='center')
def arrow(a,b,color='#52708e',dash=False,label=None):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.5,color=color,linestyle='--' if dash else '-'))
    if label:txt((a[0]+b[0])/2+.07,(a[1]+b[1])/2,label,9,color)
def lane(y,h,title,color):
    ax.add_patch(FancyBboxPatch((.2,y),16.6,h,boxstyle='round,pad=0.01,rounding_size=.16',facecolor=color,edgecolor='none'))
    txt(.5,y+h-.3,title,13,weight='bold')
txt(.35,11.04,'Depth latent를 이용한 단차·지면 보행 학습 — GREENY 제안 구조',21,weight='bold')
txt(.35,10.6,'논문 [1–4]의 아이디어를 조합한 설계 예시 · 학습 코드 미구현 · 수치와 보조 손실은 프로젝트 제안',10,color='#60748d')
lane(7.55,2.65,'01  Teacher 학습  |  시뮬레이션에서만 사용 · PPO [3]', '#eaf1fd')
lane(4.65,2.55,'02  Depth student 학습  |  teacher를 고정하고 모방 · DAgger [4]', '#e8f6f2')
lane(1.75,2.5,'03  실행  |  teacher / 정답 지형 / critic 없이 student만 사용', '#f0edf9')
xs=[.55,3.9,7.25,10.6,13.95];w=2.5;h=1.02
for x,t,b in zip(xs,['MuJoCo 환경','지형 높이 샘플','Terrain encoder','Teacher latent','Teacher policy'],['자유 몸체 + 실제 접촉\n평지 → 단차 → 계단','로봇 기준 주변·전방 지형\n학습 전용 정답 H_t','MLP\n보행 보상으로 함께 학습','z_T = E_T(H_t)\n예: 64차원','입력: z_T, 관절·IMU, 명령\n출력: 관절각 보정 a_T']):box(x,8.6,w,h,t,b)
for i in range(4):arrow((xs[i]+w,9.11),(xs[i+1],9.11))
txt(.65,7.96,'환경에서 행동 실행 → 보상·종료·rollout → PPO로 encoder / actor / critic 갱신',11)
txt(.65,7.68,'critic은 정답 상태 사용 가능 · 학습 후 제거',9,color='#60748d')
for x,t,b in zip(xs,['원본 Depth','Depth CNN','시간 정보 통합','Student latent','Student policy'],['D_t [m] + 유효성 mask\n컬러맵·HUD 제외','정규화 / resize\n예: 96×72, 2채널','GRU(CNN, 관절·IMU, 명령)\n이전 hidden state 유지','z_S ≈ z_T\n예: 64차원','teacher actor에서 초기화\n출력: 관절각 보정 a_S']):box(x,5.35,w,h,t,b,edge='#92c5b4')
for i in range(4):arrow((xs[i]+w,5.86),(xs[i+1],5.86),color='#34856f')
arrow((11.85,8.59),(11.85,6.38),color='#9d65ac',dash=True,label='latent 지도')
arrow((15.2,8.59),(15.2,6.38),color='#9d65ac',dash=True,label='행동 지도')
txt(.65,4.95,'Student rollout의 같은 상태에서 teacher 정답 생성 → 행동 모방 + latent 정렬; 지형 복원 보조 head는 선택 [1,2,4]',10)
for x,t,b in zip(xs,['관측','인식 + 기억','보행 정책','관절 제어','로봇 / MuJoCo'],['Depth + mask + timestamp\n관절·IMU + 속도 명령','CNN → GRU → z_S\n예: 영상 10–20 Hz','MLP(z_S, 관측, 명령)\n예: 정책 50 Hz','q_target = q_nom + scale·a\nPD + 관절·토크 제한','몸체 자유 운동\n지면 접촉 / 단차 통과']):box(x,2.55,w,h,t,b,edge='#b6a7d1')
for i in range(4):arrow((xs[i]+w,3.06),(xs[i+1],3.06),color='#81709d')
ax.plot([15.2,15.2,1.8,1.8],[2.53,2.05,2.05,2.53],color='#81709d',lw=1.2)
arrow((1.8,2.12),(1.8,2.55),color='#81709d')
txt(8.5,2.25,'다음 시점 센서 피드백 · reset 시 GRU 메모리 초기화',9,color='#655183',ha='center')
txt(.45,1.3,'설계 조건: 지형은 보행에 필요한 범위만 지도 · 카메라 각도 고정부터 시작 · 지연/결측/마찰을 학습 중 변동',10)
txt(.45,.87,'[1] Agarwal et al., CoRL 2022: Egocentric Vision   [2] Miki et al., Science Robotics 2022: Robust Perceptive Locomotion',9,color='#60748d')
txt(.45,.52,'[3] Schulman et al., 2017: PPO   [4] Ross et al., AISTATS 2011: DAgger   |   전체 서지·링크·손실 정의: depth_locomotion_framework.md',9,color='#60748d')
fig.subplots_adjust(left=.015,right=.99,top=.995,bottom=.015)
for ext in ('svg','png','pdf'):fig.savefig('greeny/docs/depth_locomotion_framework.'+ext,facecolor=fig.get_facecolor())
