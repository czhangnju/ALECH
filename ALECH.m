function [B1, B2, B3] = ALECH(X, Y, L, param, XTest, YTest)

% hash code learning
% min ||WV-PL||^2 + ||L-PL||^2 + delta*||B-V||^2 + theta*||rG'*G-V'B||^2 + eta*R(W,P)
% s.t. VV'=nI_r, V1=0, B\in{-1,1}.
%
% hash function learning
% min ||B-Wi*Xi||^2 + lamb*||B'*(Wi*Xi)-r*S||^2
%
G= NormalizeFea(L,1);
[n, dX] = size(X);
dY = size(Y,2);
X = X'; Y=Y'; L=L';G=G';
c = size(L,1);

nbits = param.nbits;
alpha = param.alpha;
beta = param.beta;
lamb = param.lamb;
eta = param.eta;
%rand('seed', 2020);
sel_sample = Y(:,randsample(n, 2000),:);
[pcaW, ~] = eigs(cov(sel_sample'), nbits);
V = pcaW'*Y;
B = sign(V);
B(B==0) = -1;
P = eye(c);
W = (P*L*V')/(V*V'+lamb*eye(nbits));
On = ones(1,n);

for iter = 1:param.iter  

   % B-step
    B = sign(2*nbits*alpha*V*G'*G - nbits*alpha*V*On'*On + beta*V);
    
    %
    ZA = 2*L*L' + eta*eye(c);
    ZB = W*V*L' + L*L';
    P = ZB/ZA;
    clear ZA ZB;

    % W-step
    W = (P*L*V')/(V*V'+ eta*eye(nbits));
    
    % V-step
     Z = W'*P*L + beta*B + 2*alpha*nbits*B*G'*G - alpha*nbits*B*On'*On;
     Z = Z';
     Temp = Z'*Z-1/n*(Z'*ones(n,1)*(ones(1,n)*Z));
     [~,Lmd,QQ] = svd(Temp); clear Temp
     idx = (diag(Lmd)>1e-4);
     Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
     Pt = (Z-1/n*ones(n,1)*(ones(1,n)*Z)) *  (Q / (sqrt(Lmd(idx,idx))));
     P_ = orth(randn(n,nbits-length(find(idx==1))));
     V = sqrt(n)*[Pt P_]*[Q Q_]';
     V = V';  

   
end
    
    B1 = B';		
	B1 = B1>0;
    Gx = pinv(B*B'+lamb*eye(nbits))*(2*nbits*B*G'*G*X' - nbits*B*On'*On*X' +lamb*B*X')*pinv(X*X');
    B2 = XTest*Gx'>0;
    Gy = pinv(B*B'+lamb*eye(nbits))*(2*nbits*B*G'*G*Y'- nbits*B*On'*On*Y'+lamb*B*Y')*pinv(Y*Y');
    B3 = YTest*Gy'>0;
    
end

