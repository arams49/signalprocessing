%% Plot of designed Aritificial Neural Network
figure;
hold on;
set(gca,'YTick',[]);
set(gca,'XTick',[]);
axis equal;
title('Artificial Neural Network Design');
pt1 = ( 2*(-(Ni-1)/2:(Ni-1)/2) )';
pt2 = ( 2*(-(Nh-1)/2:(Nh-1)/2) )';
pt3 = ( 2*(-(No-1)/2:(No-1)/2) )';

Ctr1 = [repmat(2,[Ni,1]),pt1];
Ctr2 = [repmat(5,[Nh,1]),pt2];
Ctr3 = [repmat(8,[No,1]),pt3];
Rd1 = repmat(0.2,[Ni,1]);
Rd2 = repmat(0.2,[Nh,1]);
Rd3 = repmat(0.2,[No,1]);
viscircles(Ctr1,Rd1,'EdgeColor','k');
viscircles(Ctr2,Rd2,'EdgeColor','k','LineStyle','--');
viscircles(Ctr3,Rd3,'EdgeColor','k');
for i = 1:Ni
    for j = 1:Nh
        line([Ctr1(i,1),Ctr2(j,1)],[Ctr1(i,2),Ctr2(j,2)]);
        txt = num2str(P(i,j));
        C = (2.5*Ctr1(i,:) + Ctr2(j,:))/3.5;
        text(C(1),C(2),txt);
    end
end
for i = 1:Nh
    for j = 1:No
        line([Ctr2(i,1),Ctr3(j,1)],[Ctr2(i,2),Ctr3(j,2)]);
        txt = num2str(Q(i,j));
        C = (2.5*Ctr2(i,:) + Ctr3(j,:))/3.5;
        text(C(1),C(2),txt);
    end
end
for i = 1:Nh
    txt = num2str(u(i));
    text(Ctr2(i,1)-0.3,Ctr2(i,2)+0.4,txt);
end
for i = 1:No
    line([Ctr3(i,1),Ctr3(i,1)+1],[Ctr3(i,2),Ctr3(i,2)]);
    txt = num2str(v(i));
    text(Ctr3(i,1)-0.3,Ctr3(i,2)+0.4,txt);
end
for i = 1:Ni
    line([Ctr1(i,1)-1,Ctr1(i,1)],[Ctr1(i,2),Ctr1(i,2)]);
end

txt = {'Input','Layer'};
text(Ctr1(1,1)-0.25,Ctr1(1,2)-0.7,txt,'FontWeight','bold');
txt = {'Hidden','Layer'};
text(Ctr2(1,1)-0.25,Ctr1(1,2)-0.7,txt,'FontWeight','bold');
txt = {'Output','Layer'};
text(Ctr3(1,1)-0.25,Ctr1(1,2)-0.7,txt,'FontWeight','bold');