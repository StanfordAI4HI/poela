% This file will read the MIMIC table data, and generate a csv file for use
% in python.

% pre-req : run fixMIMICtavle.m in data folder before running this.

%% Section 1 : Read data
load('MIMICtable.mat')

colbin = {'gender','mechvent','max_dose_vaso','re_admission'};
colnorm={'age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',...
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',...
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',...
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index',...
    'PaO2_FiO2','cumulated_balance'};
collog={'SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR',...
    'input_total','input_4hourly','output_total','output_4hourly'};

colbin=find(ismember(MIMICtable.Properties.VariableNames,colbin));colnorm=find(ismember(MIMICtable.Properties.VariableNames,colnorm));collog=find(ismember(MIMICtable.Properties.VariableNames,collog));

% find patients who died in ICU during data collection period
% ii=MIMICtable.bloc==1&MIMICtable.died_within_48h_of_out_time==1& MIMICtable.delay_end_of_record_and_discharge_or_death<24;
% icustayidlist=MIMICtable.icustayid;
% ikeep=~ismember(icustayidlist,MIMICtable.icustayid(ii));
reformat5=table2array(MIMICtable);
% reformat5=reformat5(ikeep,:);
icustayidlist=MIMICtable.icustayid;
icuuniqueids=unique(icustayidlist); %list of unique icustayids from MIMIC

MIMICraw=MIMICtable(:, [colbin colnorm collog]);
MIMICraw=table2array(MIMICraw);  % RAW values
MIMICzs=[reformat5(:, colbin)-0.5 zscore(reformat5(:,colnorm)) zscore(log(0.1+reformat5(:, collog)))];  
MIMICzs(:,[4])=log(MIMICzs(:,[ 4])+.6);   % MAX DOSE NORAD
% deleted next line 45 : input hourly 4
% MIMICzs(:,45)=2.*MIMICzs(:,45);   % increase weight of this variable
%% Section 2 : Create Actions
nact=5^2; % number of actions = 5
 
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
 
a= reformat5(:,iol);                   %IV fluid
a= tiedrank(a(a>0)) / length(a(a>0));   % excludes zero fluid (will be action 1)
 
iof=floor((a+0.2499999999)*4);  %converts iv volume in 4 actions
a= reformat5(:,iol); a=find(a>0);  %location of non-zero fluid in big matrix
io=ones(size(reformat5,1),1);  %array of ones, by default     
io(a)=iof+1;   %where more than zero fluid given: save actual action
vc=reformat5(:,vcl);  vcr= tiedrank(vc(vc~=0)) / numel(vc(vc~=0)); vcr=floor((vcr+0.249999999999)*4);  %converts to 4 bins
vcr(vcr==0)=1; vc(vc~=0)=vcr+1; vc(vc==0)=1;
ma1=[ median(reformat5(io==1,iol))  median(reformat5(io==2,iol))  median(reformat5(io==3,iol))  median(reformat5(io==4,iol))  median(reformat5(io==5,iol))];  %median dose of drug in all bins
ma2=[ median(reformat5(vc==1,vcl))  median(reformat5(vc==2,vcl))  median(reformat5(vc==3,vcl))  median(reformat5(vc==4,vcl))  median(reformat5(vc==5,vcl))] ;

med=[io vc];
[uniqueValues,~,actionbloc] = unique(array2table(med),'rows');

uniqueValuesdose=[ ma2(uniqueValues.med2)' ma1(uniqueValues.med1)'];  % median dose of each bin for all 25 actions
 
%% Section 3 : Create Reward
outcome = 10;
Y90=reformat5(:,outcome); 

%% Final Data
step=reformat5(:,1);
patient_id=reformat5(:,2);

data = [patient_id, step, MIMICzs actionbloc Y90];
% 4; % max vasso
% 45; % input_4hourly
% 44; % input_4hourly total vasso
data(:, [4, 45, 44]) = [];
% outcome 1 = died
variable_names = {'id', 'step', 'gender', 're_admission', 'mechvent','age', 'Weight_kg', 'GCS', ...
                  'HR', 'SysBP', 'MeanBP', 'DiaBP', 'RR', 'Temp_C', 'FiO2_1', 'Potassium', ...
                  'Sodium', 'Chloride', 'Glucose', 'Magnesium', 'Calcium', 'Hb', 'WBC_count', ...
                  'Platelets_count', 'PTT', 'PT', 'Arterial_pH', 'paO2', 'paCO2', 'Arterial_BE', ...
                  'Arterial_lactate', 'HCO3', 'Shock_Index', 'PaO2_FiO2', 'cumulated_balance', ...
                  'SOFA', 'SIRS', 'SpO2', 'BUN', 'Creatinine', 'SGOT', 'SGPT', 'Total_bili', ...
                  'INR', 'output_total', 'output_4hourly', 'action', 'outcome_if_died'};
data = array2table(data, 'VariableNames', variable_names);
writetable(data,"MIMIC_outcome.csv");