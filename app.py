import streamlit as st
import pickle as pkl
import pandas as pd

st.set_page_config(layout='wide')
st.title('IPL Winning Prediction')

Model=pkl.load(open('Model.pkl','rb'))
batsman_stat=pkl.load(open('batsman_stat.pkl','rb'))
bowlers_stat=pkl.load(open('bowlers_stat.pkl','rb'))
toss_decision=pkl.load(open('Toss_decision.plk','rb'))
teams=pkl.load(open('teams.pkl','rb'))
venue=pkl.load(open('venue.pkl','rb'))
season=pkl.load(open('season.pkl','rb'))

col1,col2,col3,col4=st.columns(4)

with col1:
    Toss_winner=st.selectbox('Select the team which won the toss',sorted(teams))
with col2:
    Toss_decision=st.selectbox('Select the toss decision',sorted(toss_decision))
with col3:
    Match_venue=st.selectbox('Select the venue of match',sorted(venue))
with col4:
    IPL_Season=st.number_input('Current season',max_value=2026,min_value=2020,step=1)    

col5,col6,col7,col8=st.columns(4)    

with col5:
    Batting_team=st.selectbox('Select the batting team',sorted(teams))
with col6:
    Bowling_team=st.selectbox('Select the bowling team',sorted(teams))
with col7:
    Score=st.number_input('Batting team score till now',max_value=720,min_value=50,step=1)
with col8:
    Target_runs=st.number_input('Total target',max_value=720,min_value=50,step=1)

col9,col10,col11=st.columns(3)    

with col9:
    Wickets_gone=st.number_input('Wickets already fell',max_value=10,min_value=0,step=1)
with col10:
    Overs_finished=st.number_input('number of completed overs',max_value=20,min_value=0,step=1)
with col11:
    balls_bowled=st.number_input('Ball of the over',max_value=6,min_value=0,step=1)    

col12,col13=st.columns(2)

with col12:
    batsman=st.selectbox('Select the batsman',sorted(list(batsman_stat.index)))
with col13:
    bowler=st.selectbox('Select the bowler',sorted(list(bowlers_stat.index)))

if st.button('PREDICT PROBABILITY'):

    batsman_sr=batsman_stat.loc[batsman,'batsman_sr']   
    economy=bowlers_stat.loc[bowler,'economy']
    Wicket_rate=bowlers_stat.loc[bowler,'wicket_rate']
    runs_left=Target_runs-Score
    Remaining_balls=120-((Overs_finished*6)+balls_bowled)
    Wickets_left=10-Wickets_gone

    total_overs = Overs_finished + (balls_bowled / 6)

    if total_overs == 0:
        CRR = 0
    else:
        CRR = Score / total_overs
    RRR=runs_left/(20-(Overs_finished+(balls_bowled/6)))
    
    input_df=pd.DataFrame({
        'toss_winner':[Toss_winner],
        'toss_decision':[Toss_decision],
        'venue':[Match_venue],
        'Season_year':[IPL_Season],
        'batting_team':[Batting_team],
       'bowling_team':[Bowling_team],
        'batsman_sr':[batsman_sr],
        'economy':[economy],
        'wicket_rate':[Wicket_rate],
        'score':[Score],
       'target_left':[runs_left],
        'wickets':[Wickets_left],
        'remaining_balls':[Remaining_balls],
        'CRR':[CRR],
        'RRR':[RRR]
    })

    result=Model.predict_proba(input_df)
    win=result[0][0]
    lose=result[0][1]

    st.header(Batting_team + ' - '+ str(round(win*100)) + '%')
    st.header(Bowling_team + ' - '+ str(round(lose*100)) + '%')
    







    

