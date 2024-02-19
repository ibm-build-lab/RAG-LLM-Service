FROM registry.access.redhat.com/ubi8/python-311:latest

WORKDIR /deployment

COPY app.py /deployment
COPY templates/* /deployment/templates/
COPY requirements.txt /deployment
COPY tls.crt /deployment
COPY .env /deployment
COPY promptJSON /deployment
COPY promptSQL /deployment
COPY db2jcc4.jar /deployment

RUN pip3 install -r requirements.txt
USER 0
RUN yum install -y java-1.8.0-openjdk.x86_64

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]
