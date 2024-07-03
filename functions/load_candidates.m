function [rA,rB,m,mB,rx,ry,indIn] = load_candidates(figFlag)

% define candidates
L = [2.4 1.6];
mx = 49;
my = 33;

% define ellipses
R1 = [0.4 0.3];
C1 = [-0.5 0];
R2 = [0.4 0.3];
C2 = [0.5 0];

[rx,ry] = meshgrid(linspace(-L(1)/2,L(1)/2,mx),...
                   linspace(-L(2)/2,L(2)/2,my));

r = [rx(:), ry(:)];

indEllip1 = ((r(:,1)-C1(1)).^2 / R1(1)^2)+((r(:,2)-C1(2)).^2 / R1(2)^2) < 1;
indEllip2 = ((r(:,1)-C2(1)).^2 / R2(1)^2)+((r(:,2)-C2(2)).^2 / R2(2)^2) < 1;
indIn = or(indEllip1,indEllip2);

rB = r(indIn,:);
rA = r(~indIn,:);
m = size(rA,1);
mB = size(rB,1);

rA = [rA, zeros(size(rA,1),1)];
rB = [rB, zeros(size(rB,1),1)];

if figFlag
    figure
    scatter(rA(:,1),rA(:,2))
    hold on
    scatter(rB(:,1),rB(:,2))
    axis equal
end

end