PYTHONUNBUFFERED=1
echo "Activate virtual environment"
source .venv/bin/activate

echo "Publish"
flit publish --repository {{input.zrb_extras_repo}}
