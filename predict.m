%function [errors,coef,w,selection,Y_train] = predict(X_train, Y_train,w1,w2,coef,ii,selection,y)  

	tol = 1e-10;   %tolerance for the factor function error,
	Y_train=[Y_train; y];
	extension = [selection ii];

	pred1 = X_train(ii,:)*w1;
	pred2 = X_train(ii,:)*w2;

	if (Y_train(ii)*pred1>=0) 
	    coef(selection) = aco1;
	else
	    coef(selection) = aco2;
	end

	coef = [coef; 0];
	selection = extension;
	preds = X_train*X_train'*coef;
	error = sum(Y_train.*preds <=0)/length(Y_train);
%	errors=[errors;error];
	w = X_train'*coef;

