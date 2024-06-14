import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Streamlit 앱 구성
st.title(":rainbow[전투 데이터 분석 및 예측 도구]")
st.sidebar.title("기능 선택")

# 사이드바 메뉴
menu = st.sidebar.selectbox("MENU", ["홈", "지형 CSV 파일 업로드 및 분석", "사단 승률 예측", "두 사단의 전투 결과 예측"])

if menu == "홈":
    st.header("Hearts of Iron IV 지형 분석 및 전투 예측 앱")
    
    st.image("https://i.imgur.com/Uzz4XI7.png", caption="전투 분석 도구")  # 이미지 경로를 실제 경로로 변경
    
elif menu == "지형 CSV 파일 업로드 및 분석":
    st.header("지형 CSV 파일 업로드 및 분석")
    # 파일 업로드
    uploaded_file = st.file_uploader("CSV 파일을 업로드", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, sep=';')
        con1 = st.container(border=True)
        con1.subheader("데이터 주요부분 추출")
        df.columns = ['PID', 'R', 'G', 'B', 'type', 'coastal', 'terrain', 'continent']
        df.drop(columns=['R', 'G', 'B'], inplace=True)
        con1.markdown("**[기존 column]**")
        con1.markdown(":red[PID]는 땅 한 칸의 :red[Province ID]를 나타낸다.")
        con1.markdown(":orange[type]는 육지/호수/바다 등의 :orange[Province 종류]를 구분하는 값이다.")
        con1.markdown(":green[coastal]은 :green[해안가]의 땅을 True/False로 구분한다.")
        con1.markdown(":blue[terrain]은 땅의 :blue[지형]을 의미하며 10가지로 구분된다.")
        con1.markdown(":violet[continent]는 땅이 위치한 :violet[대륙]을 표시한다.")
        con1.markdown("1은 유럽, 2는 북미, 3은 남미, 4는 오세아니아, 5는 아프리카, 6은 아시아, 7은 중동에 해당한다.")

        df = df[df['type']=='land'].reset_index().drop(columns=['index'])
        df['width1'] = df['terrain']
        df['width1'].replace(to_replace='forest', value=60, inplace=True)
        df['width1'].replace(to_replace='desert', value=70, inplace=True)
        df['width1'].replace(to_replace='hills', value=70, inplace=True)
        df['width1'].replace(to_replace='jungle', value=60, inplace=True)
        df['width1'].replace(to_replace='marsh', value=50, inplace=True)
        df['width1'].replace(to_replace='mountain', value=50, inplace=True)
        df['width1'].replace(to_replace='plains', value=70, inplace=True)
        df['width1'].replace(to_replace='urban', value=80, inplace=True)
        df['width2'] = df['width1'].map(lambda x: int(1.5*x))
        df['width3'] = df['width1'].map(lambda x: 2*x)
        df['width4'] = df['width1'].map(lambda x: int(2.5*x))
        df['width5'] = df['width1'].map(lambda x: 3*x)
        df['attack'] = df['terrain']
        df['attack'].replace(to_replace='forest', value=-15, inplace=True)
        df['attack'].replace(to_replace='desert', value=0, inplace=True)
        df['attack'].replace(to_replace='hills', value=-25, inplace=True)
        df['attack'].replace(to_replace='jungle', value=-30, inplace=True)
        df['attack'].replace(to_replace='marsh', value=-40, inplace=True)
        df['attack'].replace(to_replace='mountain', value=-50, inplace=True)
        df['attack'].replace(to_replace='plains', value=0, inplace=True)
        df['attack'].replace(to_replace='urban', value=-30, inplace=True)
        con1.dataframe(df)
        width_values = pd.concat([df['width1'], df['width2'], df['width3'], df['width4'], df['width5']])
        con1.markdown("**[새로 추가한 column]**")
        con1.markdown("width1은 각 terrain에 따른 전장너비 값을 수치로 치환한 것이다.")
        con1.markdown("width2~5는 다방면에서 공격할 시 전장너비의 절반씩 더해지는 것을 고려한 값이다.")
        con1.markdown(":red[attack]은 각 terrain에 따른 :red[기본 전투 페널티]를 나타낸다.")

        # 각 수치가 몇 번씩 나오는지 계산
        value_counts = width_values.value_counts().sort_index()
        sorted_dict = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_keys = [key for key, value in sorted_dict]
        sorted_keys = sorted_keys[:15]
        sorted_values = [value for key, value in sorted_dict]
        sorted_values = sorted_values[:15]

        # 그래프 그리기
        con2 = st.container(border=True)
        con2.subheader("전장너비 데이터 시각화")
        plt.figure(figsize=(10, 6))
        bars = value_counts.plot(kind='bar', color='#4f6578')
        # bars.patches[2].set_color('#ba3e41')
        # bars.patches[7].set_color('#ba3e41')
        # bars.patches[10].set_color('#ba3e41')
        # bars.patches[11].set_color('#1f77b4')
        # bars.patches[8].set_color('#ff8439')
        plt.xlabel('Width Values')
        plt.ylabel('Frequency')
        plt.title('Number of battlefield widths around the world')
        con2.pyplot(plt)        
        con2.markdown("width1~5 데이터를 이용해 각 전장너비 값이 전세계에 얼마나 존재하는지 그래프로 표현했다.")
        con2.markdown("hoi4의 기본 땅 데이터에서는 **:red[70, 105, 140] / :blue[150] / :orange[120]** 등이 주목할 만한 전장너비이다.")

        con3 = st.container(border=True)
        con3.subheader("사단 편제의 전장너비 별 받는 페널티의 양")

        penalty_li = [[0 for _ in range(15)] for _ in range(50)]

        for i in range(1, 51):
            for index, j in enumerate(sorted_keys):
                if j%i:
                    if (i - (j % i))*-2.5 - int((i - (j % i))*-2.5) == 0:
                        penalty_li[i-1][index] = int((i - (j % i))*-2.5)
                    else:
                        penalty_li[i-1][index] = (i - (j % i))*-2.5
                else:
                    penalty_li[i-1][index] = 0
        df_penalty = pd.DataFrame(penalty_li)
        
        for i in range(15):
            df_penalty[i] = df_penalty[i].map(lambda x: x*sorted_values[i]*100/sum(sorted_values)*0.001)

        df_penalty['sum'] = df_penalty[0] + df_penalty[1] + df_penalty[2] + df_penalty[3] + df_penalty[4] + df_penalty[5] + df_penalty[6] + df_penalty[7] + df_penalty[8] + df_penalty[9] + df_penalty[10] + df_penalty[11] + df_penalty[12] + df_penalty[13] + df_penalty[14]
        plt.figure(figsize=(10, 6))
        plt.plot(df_penalty.index+1, df_penalty['sum'], marker='o', linestyle='-')
        plt.xlabel('Width Values')
        plt.ylabel('Value')
        plt.grid(True)
        con3.pyplot(plt)
        con3.markdown("사단 편제의 전장너비에 따라 받는 **전장너비 초과 페널티**의 양을 나타낸 그래프이다.")
        con3.markdown("전장너비를 비우지 않고 :red[최대한 채우는 상태]를 기준으로 고려했다.")
        con3.markdown("이를테면, 70너비의 전장에서 20너비의 사단은 총 4사단이 전투에 참여하게 된다.")
        con3.write("")
        con3.markdown("전장너비의 빈도 수에 따라 :blue[가중치]를 부여하여 합산 페널티를 계산해 그래프 상에 나타냈다.")
        con3.markdown("그래프 상에서 위쪽에 위치하는 점들이 페널티가 적은 점들이다.")
        con3.markdown("hoi4의 기본 땅 데이터에서는 **5w, 10w, 15w, 18w, 20w, 25w, 30w, 35w > 36w, 38w** 순으로 유용하다.")


elif menu == "사단 승률 예측":

    # st.header("전투 샘플 데이터 업로드")
    sample_data = 1

    if sample_data is not None:
        combat = pd.read_csv('combat.csv')
        combat.drop(columns=['Infantry', 'Artillery', 'Anti-Air', 'Anti-Tank', 'Opp-IN', 'Opp-AR', 'Opp-AA', 'Opp-AT'], inplace=True)

        MAX_HP = 0
        if max(combat['HP']) >= max(combat['Opp-HP']):
            MAX_HP = max(combat['HP'])
        else:
            MAX_HP = max(combat['Opp-HP'])

        MIN_HP = 0
        if min(combat['HP']) <= min(combat['Opp-HP']):
            MIN_HP = min(combat['HP'])
        else:
            MIN_HP = min(combat['Opp-HP'])

        MAX_Org = 0
        if max(combat['Org.']) >= max(combat['Opp-O']):
            MAX_Org = max(combat['Org.'])
        else:
            MAX_Org = max(combat['Opp-O'])

        MIN_Org = 0
        if min(combat['Org.']) <= min(combat['Opp-O']):
            MIN_Org = min(combat['Org.'])
        else:
            MIN_Org = min(combat['Opp-O'])

        MAX_Soft = 0
        if max(combat['Soft']) >= max(combat['Opp-S']):
            MAX_Soft = max(combat['Soft'])
        else:
            MAX_Soft = max(combat['Opp-S'])

        MIN_Soft = 0
        if min(combat['Soft']) <= min(combat['Opp-S']):
            MIN_Soft = min(combat['Soft'])
        else:
            MIN_Soft = min(combat['Opp-S'])

        MAX_Hard = 0
        if max(combat['Hard']) >= max(combat['Opp-H']):
            MAX_Hard = max(combat['Hard'])
        else:
            MAX_Hard = max(combat['Opp-H'])

        MIN_Hard = 0
        if min(combat['Hard']) <= min(combat['Opp-H']):
            MIN_Hard = min(combat['Hard'])
        else:
            MIN_Hard = min(combat['Opp-H'])

        MAX_Def = 0
        if max(combat['Defence']) >= max(combat['Opp-D']):
            MAX_Def = max(combat['Defence'])
        else:
            MAX_Def = max(combat['Opp-D'])

        MIN_Def = 0
        if min(combat['Defence']) <= min(combat['Opp-D']):
            MIN_Def = min(combat['Defence'])
        else:
            MIN_Def = min(combat['Opp-D'])

        MAX_Br = 0
        if max(combat['Breakthr.']) >= max(combat['Opp-B']):
            MAX_Br = max(combat['Breakthr.'])
        else:
            MAX_Br = max(combat['Opp-B'])

        MIN_Br = 0
        if min(combat['Breakthr.']) <= min(combat['Opp-B']):
            MIN_Br = min(combat['Breakthr.'])
        else:
            MIN_Br = min(combat['Opp-B'])

        MAX_W = 0
        if max(combat['Width']) >= max(combat['Opp-W']):
            MAX_W = max(combat['Width'])
        else:
            MAX_W = max(combat['Opp-W'])

        MIN_W = 0
        if min(combat['Width']) <= min(combat['Opp-W']):
            MIN_W = min(combat['Width'])
        else:
            MIN_W = min(combat['Opp-W'])

        combat['HP'] = combat['HP'].map(lambda x: (x-MIN_HP)/(MAX_HP-MIN_HP))
        combat['Org.'] = combat['Org.'].map(lambda x: (x-MIN_Org)/(MAX_Org-MIN_Org))
        combat['Soft'] = combat['Soft'].map(lambda x: (x-MIN_Soft)/(MAX_Soft-MIN_Soft))
        combat['Hard'] = combat['Hard'].map(lambda x: (x-MIN_Hard)/(MAX_Hard-MIN_Hard))
        combat['Defence'] = combat['Defence'].map(lambda x: (x-MIN_Def)/(MAX_Def-MIN_Def))
        combat['Breakthr.'] = combat['Breakthr.'].map(lambda x: (x-MIN_Br)/(MAX_Br-MIN_Br))
        combat['Width'] = combat['Width'].map(lambda x: (x-MIN_W)/(MAX_W-MIN_W))
        combat['Opp-HP'] = combat['Opp-HP'].map(lambda x: (x-MIN_HP)/(MAX_HP-MIN_HP))
        combat['Opp-O'] = combat['Opp-O'].map(lambda x: (x-MIN_Org)/(MAX_Org-MIN_Org))
        combat['Opp-S'] = combat['Opp-S'].map(lambda x: (x-MIN_Soft)/(MAX_Soft-MIN_Soft))
        combat['Opp-H'] = combat['Opp-H'].map(lambda x: (x-MIN_Hard)/(MAX_Hard-MIN_Hard))
        combat['Opp-D'] = combat['Defence'].map(lambda x: (x-MIN_Def)/(MAX_Def-MIN_Def))
        combat['Opp-B'] = combat['Opp-B'].map(lambda x: (x-MIN_Br)/(MAX_Br-MIN_Br))
        combat['Opp-W'] = combat['Opp-W'].map(lambda x: (x-MIN_W)/(MAX_W-MIN_W))

        X = combat.drop(columns=['Winner'])
        y = combat['Winner']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 데이터 전처리 (스케일링)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 모델 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # 예측
        y_pred = model.predict(X_test)

        # 모델 평가
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"[머신러닝 정확도: {accuracy}]")

        con = st.container(border=True)
        con.header("군대 승률 예측")
        
        con.write("사단의 편제를 입력")
        feature1 = con.number_input("Infantry", min_value=0, max_value=25, step=1)
        feature2 = con.number_input("Artillery", min_value=0, max_value=25, step=1)
        art = feature2
        feature3 = con.number_input("Anti-Air", min_value=0, max_value=25, step=1)
        aa = feature3
        feature4 = con.number_input("Anti-Tank", min_value=0, max_value=25, step=1)
        at = feature4
        if feature1+feature2+feature3+feature4:
            inf = feature1
        else:
            inf = 1
        hp = 25 * inf + 0.6 * (art + aa + at)
        org = 60 * (inf / (inf + art + aa + at))
        soft = 6 * inf + 25 * art + 3 * aa + 4 * at
        hard = inf + 2 * art + 7 * aa + 20 * at
        de = 22 * inf + 10 * art + 4 * (aa + at)
        br = 3 * inf + 6 * art + aa
        w = 2 * inf + 3 * art + aa + at

        hp = (hp-MIN_HP)/(MAX_HP-MIN_HP)
        org = (org-MIN_Org)/(MAX_Org-MIN_Org)
        soft = (soft-MIN_Soft)/(MAX_Soft-MIN_Soft)
        hard = (hard-MIN_Hard)/(MAX_Hard-MIN_Hard)
        de = (de-MIN_Def)/(MAX_Def-MIN_Def)
        br = (br-MIN_Br)/(MAX_Br-MIN_Br)
        w = (w-MIN_W)/(MAX_W-MIN_W)
        
        test_result = []
        for i in range(0, 200):
            test_data = pd.DataFrame({
                'HP': [hp],
                'Org.': [org],
                'Soft': [soft],
                'Hard': [hard],
                'Defence': [de],
                'Breakthr.': [br],
                'Width': [w],
                'IsAttacker': [1],
                'Opp-HP': [combat['HP'][i*2]],
                'Opp-O': [combat['Org.'][i*2]],
                'Opp-S': [combat['Soft'][i*2]],
                'Opp-H': [combat['Hard'][i*2]],
                'Opp-D': [combat['Defence'][i*2]],
                'Opp-B': [combat['Breakthr.'][i*2]],
                'Opp-W': [combat['Width'][i*2]],
                'IsAttacker.1': [0],
                })
            test_data = scaler.transform(test_data)
            predictions = model.predict(test_data)
            if predictions[0] == 'G':
                test_result.append(1)
            else:
                test_result.append(0)

        for i in range(0, 200):
            test_data = pd.DataFrame({
                'HP': [combat['HP'][i*2]],
                'Org.': [combat['Org.'][i*2]],
                'Soft': [combat['Soft'][i*2]],
                'Hard': [combat['Hard'][i*2]],
                'Defence': [combat['Defence'][i*2]],
                'Breakthr.': [combat['Breakthr.'][i*2]],
                'Width': [combat['Width'][i*2]],
                'IsAttacker': [1],
                'Opp-HP': [hp],
                'Opp-O': [org],
                'Opp-S': [soft],
                'Opp-H': [hard],
                'Opp-D': [de],
                'Opp-B': [br],
                'Opp-W': [w],
                'IsAttacker.1': [0],
                })
            test_data = scaler.transform(test_data)
            predictions = model.predict(test_data)
            if predictions[0] == 'F':
                test_result.append(2)
            else:
                test_result.append(3)
        
        # if 'first_button_clicked' not in st.session_state:
        #     st.session_state.first_button_clicked = False
        # # 두 번째 버튼 상태를 저장할 변수
        # if 'second_button_clicked' not in st.session_state:
        #     st.session_state.second_button_clicked = False

        # msg_container = st.empty()
        # msg = f"전체 승리 확률: {(test_result.count(1)+test_result.count(2))*100/len(test_result)}%"
        # first_msg = f"공격 상황 승리 확률: {test_result.count(1)*100/(test_result.count(1)+test_result.count(0))}%"
        # second_msg = f"방어 상황 승리 확률: {test_result.count(2)*100/(test_result.count(2)+test_result.count(3))}%"

        # # 첫 번째 버튼
        # if st.button('승률 예측'):
        #     if feature1+feature2+feature3+feature4:
        #         st.session_state.first_button_clicked = True
        #         with msg_container.container(border = True):
        #             st.write(msg)
        #     else:
        #         st.markdown('**:red[ERROR: 사단의 편제를 입력하세요.]**')

        # # 첫 번째 버튼이 클릭된 경우 두 번째 버튼 표시
        # if st.session_state.first_button_clicked:
        #     if st.button('세부 승률 확인'):
        #         st.session_state.second_button_clicked = True

        # # 두 번째 버튼이 클릭된 경우 메시지 출력
        # if st.session_state.second_button_clicked:
        #     with msg_container.container(border = True):
        #         st.write(msg)
        #         st.write("")
        #         st.write(first_msg)
        #         st.write(second_msg)

        # if st.button('리셋'):
        #     del st.session_state.first_button_clicked
        #     del st.session_state.second_button_clicked
        #     st.rerun()

        if st.button("승률 예측"):
            if feature1+feature2+feature3+feature4:
                st.write(f"전체 승리 확률: {(test_result.count(1)+test_result.count(2))*100/len(test_result)}%")
                st.write(f"공격 상황 승리 확률: {test_result.count(1)*100/(test_result.count(1)+test_result.count(0))}%")
                st.write(f"방어 상황 승리 확률: {test_result.count(2)*100/(test_result.count(2)+test_result.count(3))}%")
            else:
                st.markdown('**:red[ERROR: 사단의 편제를 입력하세요.]**')

            
elif menu == "두 사단의 전투 결과 예측":

    # st.header("전투 샘플 데이터 업로드")
    sample_data = 1

    if sample_data is not None:
        combat = pd.read_csv('combat.csv')
        combat.drop(columns=['Infantry', 'Artillery', 'Anti-Air', 'Anti-Tank', 'Opp-IN', 'Opp-AR', 'Opp-AA', 'Opp-AT'], inplace=True)

        MAX_HP = 0
        if max(combat['HP']) >= max(combat['Opp-HP']):
            MAX_HP = max(combat['HP'])
        else:
            MAX_HP = max(combat['Opp-HP'])

        MIN_HP = 0
        if min(combat['HP']) <= min(combat['Opp-HP']):
            MIN_HP = min(combat['HP'])
        else:
            MIN_HP = min(combat['Opp-HP'])

        MAX_Org = 0
        if max(combat['Org.']) >= max(combat['Opp-O']):
            MAX_Org = max(combat['Org.'])
        else:
            MAX_Org = max(combat['Opp-O'])

        MIN_Org = 0
        if min(combat['Org.']) <= min(combat['Opp-O']):
            MIN_Org = min(combat['Org.'])
        else:
            MIN_Org = min(combat['Opp-O'])

        MAX_Soft = 0
        if max(combat['Soft']) >= max(combat['Opp-S']):
            MAX_Soft = max(combat['Soft'])
        else:
            MAX_Soft = max(combat['Opp-S'])

        MIN_Soft = 0
        if min(combat['Soft']) <= min(combat['Opp-S']):
            MIN_Soft = min(combat['Soft'])
        else:
            MIN_Soft = min(combat['Opp-S'])

        MAX_Hard = 0
        if max(combat['Hard']) >= max(combat['Opp-H']):
            MAX_Hard = max(combat['Hard'])
        else:
            MAX_Hard = max(combat['Opp-H'])

        MIN_Hard = 0
        if min(combat['Hard']) <= min(combat['Opp-H']):
            MIN_Hard = min(combat['Hard'])
        else:
            MIN_Hard = min(combat['Opp-H'])

        MAX_Def = 0
        if max(combat['Defence']) >= max(combat['Opp-D']):
            MAX_Def = max(combat['Defence'])
        else:
            MAX_Def = max(combat['Opp-D'])

        MIN_Def = 0
        if min(combat['Defence']) <= min(combat['Opp-D']):
            MIN_Def = min(combat['Defence'])
        else:
            MIN_Def = min(combat['Opp-D'])

        MAX_Br = 0
        if max(combat['Breakthr.']) >= max(combat['Opp-B']):
            MAX_Br = max(combat['Breakthr.'])
        else:
            MAX_Br = max(combat['Opp-B'])

        MIN_Br = 0
        if min(combat['Breakthr.']) <= min(combat['Opp-B']):
            MIN_Br = min(combat['Breakthr.'])
        else:
            MIN_Br = min(combat['Opp-B'])

        MAX_W = 0
        if max(combat['Width']) >= max(combat['Opp-W']):
            MAX_W = max(combat['Width'])
        else:
            MAX_W = max(combat['Opp-W'])

        MIN_W = 0
        if min(combat['Width']) <= min(combat['Opp-W']):
            MIN_W = min(combat['Width'])
        else:
            MIN_W = min(combat['Opp-W'])

        combat['HP'] = combat['HP'].map(lambda x: (x-MIN_HP)/(MAX_HP-MIN_HP))
        combat['Org.'] = combat['Org.'].map(lambda x: (x-MIN_Org)/(MAX_Org-MIN_Org))
        combat['Soft'] = combat['Soft'].map(lambda x: (x-MIN_Soft)/(MAX_Soft-MIN_Soft))
        combat['Hard'] = combat['Hard'].map(lambda x: (x-MIN_Hard)/(MAX_Hard-MIN_Hard))
        combat['Defence'] = combat['Defence'].map(lambda x: (x-MIN_Def)/(MAX_Def-MIN_Def))
        combat['Breakthr.'] = combat['Breakthr.'].map(lambda x: (x-MIN_Br)/(MAX_Br-MIN_Br))
        combat['Width'] = combat['Width'].map(lambda x: (x-MIN_W)/(MAX_W-MIN_W))
        combat['Opp-HP'] = combat['Opp-HP'].map(lambda x: (x-MIN_HP)/(MAX_HP-MIN_HP))
        combat['Opp-O'] = combat['Opp-O'].map(lambda x: (x-MIN_Org)/(MAX_Org-MIN_Org))
        combat['Opp-S'] = combat['Opp-S'].map(lambda x: (x-MIN_Soft)/(MAX_Soft-MIN_Soft))
        combat['Opp-H'] = combat['Opp-H'].map(lambda x: (x-MIN_Hard)/(MAX_Hard-MIN_Hard))
        combat['Opp-D'] = combat['Defence'].map(lambda x: (x-MIN_Def)/(MAX_Def-MIN_Def))
        combat['Opp-B'] = combat['Opp-B'].map(lambda x: (x-MIN_Br)/(MAX_Br-MIN_Br))
        combat['Opp-W'] = combat['Opp-W'].map(lambda x: (x-MIN_W)/(MAX_W-MIN_W))

        X = combat.drop(columns=['Winner'])
        y = combat['Winner']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 데이터 전처리 (스케일링)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 모델 학습
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # 예측
        y_pred = model.predict(X_test)

        # 모델 평가
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"머신러닝 정확도: {accuracy}")

        st.header("두 사단의 전투 결과 예측")
        
        concon = st.container(border=True)
        concon.write("첫 번째 사단의 편제를 입력 (공격자)")
        feature1_1 = concon.number_input("Infantry", min_value=0, max_value=25, step=1)
        feature1_2 = concon.number_input("Artillery", min_value=0, max_value=25, step=1)
        feature1_3 = concon.number_input("Anti-Air", min_value=0, max_value=25, step=1)
        feature1_4 = concon.number_input("Anti-Tank", min_value=0, max_value=25, step=1)
        
        inf = feature1_1
        art = feature1_2
        aa = feature1_3
        at = feature1_4

        
        
        conc = st.container(border=True)
        conc.write("두 번째 사단의 편제를 입력 (방어자)")
        feature2_1 = conc.number_input("Infantry ", min_value=0, max_value=25, step=1)
        feature2_2 = conc.number_input("Artillery ", min_value=0, max_value=25, step=1)
        feature2_3 = conc.number_input("Anti-Air ", min_value=0, max_value=25, step=1)
        feature2_4 = conc.number_input("Anti-Tank ", min_value=0, max_value=25, step=1)

        inf2 = feature2_1
        art2 = feature2_2
        aa2 = feature2_3
        at2 = feature2_4

        
        
        if st.button("전투 결과 예측"):
            if inf+art+aa+at and inf2+art2+aa2+at2:
                hp = 25 * inf + 0.6 * (art + aa + at)
                org = 60 * (inf / (inf + art + aa + at))
                soft = 6 * inf + 25 * art + 3 * aa + 4 * at
                hard = inf + 2 * art + 7 * aa + 20 * at
                de = 22 * inf + 10 * art + 4 * (aa + at)
                br = 3 * inf + 6 * art + aa
                w = 2 * inf + 3 * art + aa + at

                hp = (hp-MIN_HP)/(MAX_HP-MIN_HP)
                org = (org-MIN_Org)/(MAX_Org-MIN_Org)
                soft = (soft-MIN_Soft)/(MAX_Soft-MIN_Soft)
                hard = (hard-MIN_Hard)/(MAX_Hard-MIN_Hard)
                de = (de-MIN_Def)/(MAX_Def-MIN_Def)
                br = (br-MIN_Br)/(MAX_Br-MIN_Br)
                w = (w-MIN_W)/(MAX_W-MIN_W)

                hp2 = 25 * inf2 + 0.6 * (art2 + aa2 + at2)
                org2 = 60 * (inf2 / (inf2 + art2 + aa2 + at2))
                soft2 = 6 * inf2 + 25 * art2 + 3 * aa2 + 4 * at2
                hard2 = inf2 + 2 * art2 + 7 * aa2 + 20 * at2
                de2 = 22 * inf2 + 10 * art2 + 4 * (aa2 + at2)
                br2 = 3 * inf2 + 6 * art2 + aa2
                w2 = 2 * inf2 + 3 * art2 + aa2 + at2

                hp2 = (hp2-MIN_HP)/(MAX_HP-MIN_HP)
                org2 = (org2-MIN_Org)/(MAX_Org-MIN_Org)
                soft2 = (soft2-MIN_Soft)/(MAX_Soft-MIN_Soft)
                hard2 = (hard2-MIN_Hard)/(MAX_Hard-MIN_Hard)
                de2 = (de2-MIN_Def)/(MAX_Def-MIN_Def)
                br2 = (br2-MIN_Br)/(MAX_Br-MIN_Br)
                w2 = (w2-MIN_W)/(MAX_W-MIN_W)

                test_data = pd.DataFrame({
                    'HP': [hp],
                    'Org.': [org],
                    'Soft': [soft],
                    'Hard': [hard],
                    'Defence': [de],
                    'Breakthr.': [br],
                    'Width': [w],
                    'IsAttacker': [1], 
                    'Opp-HP': [hp2],
                    'Opp-O': [org2],
                    'Opp-S': [soft2],
                    'Opp-H': [hard2],
                    'Opp-D': [de2],
                    'Opp-B': [br2],
                    'Opp-W': [w2],
                    'IsAttacker.1': [0],
                })
                test_data = scaler.transform(test_data)
                predictions = model.predict(test_data)

                if predictions[0] == 'G':
                    st.write('첫 번째 사단이 공격 성공, 두 번째 사단이 방어 실패')
                else:
                    st.write('첫 번째 사단이 공격 실패, 두 번째 사단이 방어 성공')

                test_data = pd.DataFrame({
                    'HP': [hp2],
                    'Org.': [org2],
                    'Soft': [soft2],
                    'Hard': [hard2],
                    'Defence': [de2],
                    'Breakthr.': [br2],
                    'Width': [w2],
                    'IsAttacker': [1],
                    'Opp-HP': [hp],
                    'Opp-O': [org],
                    'Opp-S': [soft],
                    'Opp-H': [hard],
                    'Opp-D': [de],
                    'Opp-B': [br],
                    'Opp-W': [w],
                    'IsAttacker.1': [0],
                })
                test_data = scaler.transform(test_data)
                predictions = model.predict(test_data)

                if predictions[0] == 'G':
                    st.write('첫 번째 사단이 방어 실패, 두 번째 사단이 공격 성공')
                else:
                    st.write('첫 번째 사단이 방어 성공, 두 번째 사단이 공격 실패')

            else:
                st.markdown('**:red[ERROR: 사단의 편제를 입력하세요.]**')
