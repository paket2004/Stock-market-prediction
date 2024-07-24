docker build -t my_ml_service api/
docker run --rm -p 5151:8080 my_ml_service
docker tag my_ml_service caxapb/mlops_project
docker push caxapb/mlops_project:latest
