#set -e

echo "🎃 Prepare playground"
sudo rm -Rf playground
cp -R playground-template playground
cd playground

echo "🎃 Activate venv"
source ../src/zrb-extras/.venv/bin/activate

export ZRB_SHOW_PROMPT=0
export PYTHONPATH=$(pwd)

echo "🎃 Add airflow"
zrb project add airflow \
    --project-dir . \
    --app-name airflow \
    --http-port 8080

echo "🎃 Add metabase"
zrb project add metabase \
    --project-dir . \
    --app-name metabase \
    --http-port 8080

echo "🎃 Add citus"
zrb project add citus \
    --project-dir . \
    --app-name citus \
    --http-port 5432

echo "🎃 Add airbyte"
zrb project add airbyte \
    --project-dir . \
    --app-name airbyte \
    --http-port 8080

echo "🎃 Playground is ready"
echo "    cd \"$(pwd)\""
echo "    source-pkg"
echo "🎃 Happy Coding :)"