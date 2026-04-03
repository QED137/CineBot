#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
COOKIE_JAR="${COOKIE_JAR:-/tmp/cinebot.cookies}"

pass_count=0
fail_count=0

pass() {
  echo "PASS: $1"
  pass_count=$((pass_count + 1))
}

fail() {
  echo "FAIL: $1"
  fail_count=$((fail_count + 1))
}

require_up() {
  code="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/health" || true)"
  if [[ "$code" != "200" ]]; then
    echo "Backend is not reachable at ${BASE_URL}. Got HTTP ${code}."
    echo "Start backend first: python -m uvicorn app_fastapi:app --host 127.0.0.1 --port 8000 --reload"
    exit 1
  fi
}

check_health() {
  code="$(curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}/api/health")"
  if [[ "$code" == "200" ]]; then pass "health endpoint"; else fail "health endpoint (HTTP ${code})"; fi
}

check_headers() {
  headers="$(curl -s -D - -o /dev/null "${BASE_URL}/api/suggestion")"
  grep -qi "x-content-type-options: nosniff" <<<"$headers" && grep -qi "x-frame-options: DENY" <<<"$headers" && grep -qi "cache-control: no-store" <<<"$headers"
  if [[ $? -eq 0 ]]; then
    pass "security headers"
  else
    fail "security headers"
  fi
}

check_query_length_cap() {
  long_msg="$(printf 'a%.0s' {1..1200})"
  code="$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/api/chat" -F "message=${long_msg}")"
  if [[ "$code" == "400" ]]; then pass "query length cap"; else fail "query length cap (HTTP ${code})"; fi
}

check_chat_rate_limit() {
  got_429=0
  for i in $(seq 1 30); do
    code="$(curl -s -o /dev/null -w "%{http_code}" -X POST "${BASE_URL}/api/chat" -F "message=rate limit test ${i}")"
    if [[ "$code" == "429" ]]; then
      got_429=1
      break
    fi
  done
  if [[ "$got_429" -eq 1 ]]; then pass "chat IP/burst rate limit"; else fail "chat IP/burst rate limit"; fi
}

check_session_budget_limit() {
  rm -f "$COOKIE_JAR"
  got_429=0
  for i in $(seq 1 20); do
    code="$(curl -s -o /dev/null -w "%{http_code}" -c "$COOKIE_JAR" -b "$COOKIE_JAR" -X POST "${BASE_URL}/api/chat" -F "message=session budget test ${i}")"
    if [[ "$code" == "429" ]]; then
      got_429=1
      break
    fi
  done
  if [[ "$got_429" -eq 1 ]]; then pass "session budget limit"; else fail "session budget limit"; fi
}

main() {
  echo "Running security smoke tests against ${BASE_URL}"
  require_up

  check_health
  check_headers
  check_query_length_cap
  check_chat_rate_limit
  check_session_budget_limit

  echo ""
  echo "Summary: ${pass_count} passed, ${fail_count} failed"
  if [[ "$fail_count" -gt 0 ]]; then
    exit 1
  fi
}

main "$@"
