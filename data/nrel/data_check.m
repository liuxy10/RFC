%[~,~,Data]=xlsread('no-rail-W2.csv'); % Walker A
%[~,~,Data]=xlsread('Walking_passive02.csv'); % Walker C
[~,~,Data]=xlsread('9_none_2.csv'); % Walker B
index= find(strcmp(Data(:,1),'Trajectories'));
raw=cell2mat(Data(index+5:end,3:119));
Clav=raw(:,19:21); Norm=[nanmean(Clav(:,1)), nanmean(Clav(:,2))]; 
Head=[(raw(:,1)+raw(:,4)+raw(:,7)+raw(:,10))/4, (raw(:,2)+raw(:,5)+raw(:,8)+raw(:,11))/4,(raw(:,3)+raw(:,6)+raw(:,9)+raw(:,12))/4]; 
Head=[Head(:,1)-Norm(1),Head(:,2)-Norm(2),Head(:,3)];
LShould=[raw(:,25)-Norm(1),raw(:,26)-Norm(2),raw(:,30)];
LElb=[raw(:,34)-Norm(1),raw(:,35)-Norm(2),raw(:,36)];
Lwrist=[(raw(:,40)+raw(:,43))/2-Norm(1),(raw(:,41)+raw(:,44))/2-Norm(2),(raw(:,42)+raw(:,45))/2];
RShould=[raw(:,49)-Norm(1),raw(:,50)-Norm(2),raw(:,51)];
RElb=[raw(:,55)-Norm(1),raw(:,56)-Norm(2),raw(:,57)];
Rwrist=[(raw(:,61)+raw(:,64))/2-Norm(1),(raw(:,62)+raw(:,65))/2-Norm(2),(raw(:,63)+raw(:,66))/2];
Pelv=[(raw(:,76)+raw(:,79))/2-Norm(1),(raw(:,80)+raw(:,74))/2+30-Norm(2),(raw(:,78)+raw(:,81))/2];
Lhip=[raw(:,70)-30-Norm(1),raw(:,71)-Norm(2),raw(:,72)-70];Rhip=[raw(:,73)+30,raw(:,74),raw(:,75)-70];
Lknee=[raw(:,85)-Norm(1),raw(:,86)-Norm(2),raw(:,87)];
Lank=[raw(:,91)-Norm(1),raw(:,92)-Norm(2),raw(:,93)];
Rknee=[raw(:,103)-Norm(1),raw(:,104)-Norm(2),raw(:,105)];
Rank=[raw(:,109)-Norm(1),raw(:,110)-Norm(2),raw(:,111)];
Fin_D=[Head,Rwrist,RElb,RShould,Clav,LShould,LElb,Lwrist,Rank,Rknee,Rhip,Pelv,Lhip,Lknee,Lank];
Footstrik= find(strcmp(Data(:,2),'Right')&strcmp(Data(:,3),'Foot Strike')); % Timeing for each steps
steps_P=Footstrik(end-21:end-1,1);
Step_T=round(cell2mat(Data(steps_P,4))*100);
TT=cell2mat(Data(index+5:end,1));
Org_D=fillgaps(Fin_D(find(TT==Step_T(1)):find(TT==Step_T(end)),:));

X = reshape(Org_D(:,1),15,3);
NX=[]
for j=1:15
            NX(1,(j-1)*3+1:j*3)=X(j,1:3);
end
        
for i=1:2000
v = VideoWriter(['C:\Users\ilee5\Desktop\biological motion\Testing videos\C_',num2str(i),'.avi']);
v.FrameRate=100;
open(v);
figure('Position', [0 0 600 420]);
for j=1:max_L
     plot(NX{i,1}(j,1:3:45),NX{i,1}(j,3:3:45),'o','MarkerSize',10,...
    'MarkerEdgeColor','y','MarkerFaceColor',[1,1,0])
     set(gca,'visible','off')
     set(gcf,'color','k')
    axis([-750 750 0 1500])
    drawnow limitrate
    walk_n(i,j)=getframe;
    writeVideo(v,getframe(gcf));
end
close(v)
close all
end


%%%%%%%2D-PCA
% for i=1:6
%   [coeffA{i,1},scoreA{i,1},latentA{i,1},~,explainA,~] = pca(BM{i,1});  
% end
% 
% for i=1:6
% rec_2d{i,1}=NX+(scoreA{i,1}(:,1)*coeffA{i,1}(:,1)');
% end
% 
% plot(rec_2d{1,1}(:,1:2:30),rec_2d{1,1}(:,2:2:30),'r','LineWidth',2)
% axis([-650 650 0 1500])
% set(gca,'xtick',[],'ytick',[])
% hold on
% plot(rec_2d{6,1}(:,1:2:30),rec_2d{6,1}(:,2:2:30),'b','LineWidth',2)