load('MIMICtable.mat')
ids = readtable('patientIDs_MIMIC3.csv');
valid_p = ismember(table2array(MIMICtable(:, 'icustayid')), table2array(ids));
MIMICtable(valid_p==0, :) = [];
save('MIMICtable.mat', 'MIMICtable');
