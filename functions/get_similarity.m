function [similarity] = get_similarity(xEst,xRef)

similarity = abs(xRef' * xEst)^2 / ((xRef'*xRef)*(xEst'*xEst));

end