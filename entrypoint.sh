#!/bin/bash
# 실행 디렉토리로 이동
cd /app

# 환경 변수로 mcp_config.json 업데이트
sed -i "s|\${GCP_BIGQUERY_KEY}|$GCP_BIGQUERY_KEY|g" mcp_config.json

# 앱 실행
exec streamlit run app_KOR.py --server.address 0.0.0.0 --server.port 8000