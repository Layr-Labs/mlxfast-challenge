#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: download-r2-object.sh OBJECT_PATH OUTPUT_PATH" >&2
  exit 2
fi

object_path="${1#/}"
output_path="$2"

# Defense in depth: the object key is a trusted workflow constant today, but
# validate it before it is signed into the SigV4 canonical request and the
# request URL so a future caller cannot smuggle path traversal (`..`/`.`
# segments), an absolute path, or control characters (a newline/CR would
# corrupt the canonical request or the HTTP request line) into the R2 request.
if [[ -z "${object_path}" ]]; then
  echo "download-r2-object: object path must not be empty" >&2
  exit 2
fi
if [[ "${object_path}" == /* ]]; then
  echo "download-r2-object: object path must not be absolute: ${object_path}" >&2
  exit 2
fi
case "/${object_path}/" in
  *"/../"*|*"/./"*)
    echo "download-r2-object: object path must not contain . or .. segments: ${object_path}" >&2
    exit 2
    ;;
esac
if [[ "${object_path}" =~ [[:cntrl:]] ]]; then
  echo "download-r2-object: object path must not contain control characters" >&2
  exit 2
fi

: "${R2_ACCESS_KEY_ID:?R2_ACCESS_KEY_ID is required}"
: "${R2_BUCKET_ENDPOINT:?R2_BUCKET_ENDPOINT is required}"
: "${R2_SECRET_ACCESS_KEY:?R2_SECRET_ACCESS_KEY is required}"

endpoint="${R2_BUCKET_ENDPOINT%/}"
case "${endpoint}" in
  https://*) ;;
  *)
    echo "download-r2-object: R2_BUCKET_ENDPOINT must be an https URL" >&2
    exit 1
    ;;
esac

endpoint_rest="${endpoint#https://}"
host="${endpoint_rest%%/*}"
base_path="${endpoint_rest#"${host}"}"
request_path="${base_path}/${object_path}"
url="https://${host}${request_path}"

region="${R2_REGION:-auto}"
service="s3"
amz_date="$(date -u +%Y%m%dT%H%M%SZ)"
date_stamp="${amz_date:0:8}"
payload_hash="$(printf '' | shasum -a 256 | awk '{print $1}')"
signed_headers="host;x-amz-content-sha256;x-amz-date"
canonical_headers="$(printf 'host:%s\nx-amz-content-sha256:%s\nx-amz-date:%s\n' "${host}" "${payload_hash}" "${amz_date}")"
canonical_request="$(printf 'GET\n%s\n\n%s\n%s\n%s' "${request_path}" "${canonical_headers}" "${signed_headers}" "${payload_hash}")"
credential_scope="${date_stamp}/${region}/${service}/aws4_request"
canonical_request_hash="$(printf '%s' "${canonical_request}" | shasum -a 256 | awk '{print $1}')"
string_to_sign="$(printf 'AWS4-HMAC-SHA256\n%s\n%s\n%s' "${amz_date}" "${credential_scope}" "${canonical_request_hash}")"

hmac_hex() {
  local key_opt="$1"
  local message="$2"
  printf '%s' "${message}" \
    | openssl dgst -sha256 -mac HMAC -macopt "${key_opt}" -binary \
    | xxd -p -c 256
}

k_date="$(hmac_hex "key:AWS4${R2_SECRET_ACCESS_KEY}" "${date_stamp}")"
k_region="$(hmac_hex "hexkey:${k_date}" "${region}")"
k_service="$(hmac_hex "hexkey:${k_region}" "${service}")"
k_signing="$(hmac_hex "hexkey:${k_service}" "aws4_request")"
signature="$(printf '%s' "${string_to_sign}" \
  | openssl dgst -sha256 -mac HMAC -macopt "hexkey:${k_signing}" \
  | awk '{print $2}')"

authorization="AWS4-HMAC-SHA256 Credential=${R2_ACCESS_KEY_ID}/${credential_scope}, SignedHeaders=${signed_headers}, Signature=${signature}"
tmp_path="${output_path}.tmp"

umask 077
mkdir -p "$(dirname "${output_path}")"
trap 'rm -f "${tmp_path}"' EXIT

download_with_aws_cli() {
  local trimmed_path="${base_path#/}"
  local bucket
  local key_prefix
  local key

  if [[ -z "${trimmed_path}" || ! -x "$(command -v aws 2>/dev/null)" ]]; then
    return 1
  fi

  bucket="${trimmed_path%%/*}"
  key_prefix="${trimmed_path#"${bucket}"}"
  key_prefix="${key_prefix#/}"
  key="${object_path}"
  if [[ -n "${key_prefix}" ]]; then
    key="${key_prefix}/${object_path}"
  fi

  echo "download-r2-object: using AWS CLI S3 path-style download"
  AWS_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID}" \
    AWS_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY}" \
    AWS_DEFAULT_REGION="${region}" \
    AWS_EC2_METADATA_DISABLED=true \
    aws \
      --endpoint-url "https://${host}" \
      s3 cp \
      "s3://${bucket}/${key}" \
      "${tmp_path}" \
      --only-show-errors \
      --no-progress
}

if download_with_aws_cli; then
  chmod 600 "${tmp_path}"
  mv "${tmp_path}" "${output_path}"
  trap - EXIT
  echo "download-r2-object: wrote ${output_path}"
  exit 0
fi

echo "download-r2-object: using signed HTTPS download"
curl \
  --fail \
  --silent \
  --show-error \
  --location \
  --retry 5 \
  --retry-all-errors \
  --retry-delay 2 \
  -H "Authorization: ${authorization}" \
  -H "x-amz-content-sha256: ${payload_hash}" \
  -H "x-amz-date: ${amz_date}" \
  --output "${tmp_path}" \
  "${url}"

chmod 600 "${tmp_path}"
mv "${tmp_path}" "${output_path}"
trap - EXIT
echo "download-r2-object: wrote ${output_path}"
