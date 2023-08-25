set -e

echo "Prepare playground"
sudo rm -Rf playground
cp -R playground-template playground
cd playground

echo "Activate Venv"
source ../src/zrb-extras/.venv/bin/activate

export ZRB_SHOW_PROMPT=0

echo "Add airflow"
zrb project add airflow \
    --project-dir . \
    --app-name airflow

echo $PYTHONPATH

echo "Add metabase"
zrb project add metabase \
    --project-dir . \
    --app-name metabase

echo "cd \"$(pwd)\""
echo "source-pkg"