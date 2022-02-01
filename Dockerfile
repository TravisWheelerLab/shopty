FROM python:3.9-slim-bullseye AS build

WORKDIR /opt
COPY Pipfile ./
COPY Pipfile.lock ./
RUN pip install pipenv \
    && pipenv sync \
    && pipenv run pip freeze > requirements.txt

FROM python:3.9-slim-bullseye

RUN mkdir /opt/shopty
WORKDIR /opt/shopty

COPY --from=build /opt/requirements.txt ./
RUN pip install -r requirements.txt
COPY shopty/ shopty/

VOLUME /data
WORKDIR /data

ENV PYTHONPATH=/opt/shopty
ENTRYPOINT ["python3", "-m", "shopty"]
