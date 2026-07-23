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

# Describe a value without revealing it: byte length plus the hex of the
# first and last byte. Enough to spot an invisible poison byte (0x0a
# newline, 0x0d CR, 0xa0 from a pasted NBSP) in a secret from the run log.
value_fingerprint() {
  local value="$1"
  local length
  length="$(LC_ALL=C printf '%s' "${value}" | wc -c | tr -d '[:space:]')"
  if [[ "${length}" == "0" ]]; then
    printf 'length=0'
    return 0
  fi
  local first_hex last_hex
  first_hex="$(LC_ALL=C printf '%s' "${value}" | head -c 1 | xxd -p | tr -d '\n')"
  last_hex="$(LC_ALL=C printf '%s' "${value}" | tail -c 1 | xxd -p | tr -d '\n')"
  printf 'length=%s first=0x%s last=0x%s' "${length}" "${first_hex}" "${last_hex}"
}

# Remove bytes that can never appear in a valid https endpoint: every ASCII
# whitespace/control byte (the trailing newline `echo | gh secret set`
# injects, CRs from Windows clipboards, tabs) plus the UTF-8 non-breaking
# space bytes (0xC2 0xA0) that rich-text editors paste. A valid endpoint is
# pure ASCII, so byte-wise deletion cannot corrupt a legitimate value; a
# poisoned-but-otherwise-correct secret is rescued instead of handing curl a
# malformed URL (curl exit 3, "URL rejected").
strip_invisible_bytes() {
  LC_ALL=C printf '%s' "$1" | LC_ALL=C tr -d '[:space:][:cntrl:]\302\240'
}

raw_bucket_endpoint="${R2_BUCKET_ENDPOINT}"
endpoint="$(strip_invisible_bytes "${raw_bucket_endpoint}")"
if [[ "${endpoint}" != "${raw_bucket_endpoint}" ]]; then
  echo "download-r2-object: stripped invisible whitespace/control bytes from R2_BUCKET_ENDPOINT ($(value_fingerprint "${raw_bucket_endpoint}"))" >&2
fi
endpoint="${endpoint%/}"
endpoint_pattern='^https://[a-z0-9.-]+(/[A-Za-z0-9._-]+)*$'
if [[ ! "${endpoint}" =~ ${endpoint_pattern} ]]; then
  echo "download-r2-object: R2_BUCKET_ENDPOINT is not a valid https R2 endpoint after sanitization ($(value_fingerprint "${raw_bucket_endpoint}")); expected https://<host>[/<bucket>[/<prefix>...]]; refusing to print the value" >&2
  exit 1
fi

# The same paste class poisons the credentials: an invisible byte inside
# R2_ACCESS_KEY_ID lands mid-string in the Authorization header (curl drops
# the malformed header and the request arrives unsigned -> R2 answers 400
# InvalidArgument: Authorization), and one inside R2_SECRET_ACCESS_KEY feeds
# the HMAC a wrong key (403 SignatureDoesNotMatch). Neither byte class is
# ever legitimate in an R2 credential, so strip and note it. The secret's
# notice reports length only - never any byte of the secret.
raw_access_key_id="${R2_ACCESS_KEY_ID}"
R2_ACCESS_KEY_ID="$(strip_invisible_bytes "${raw_access_key_id}")"
if [[ "${R2_ACCESS_KEY_ID}" != "${raw_access_key_id}" ]]; then
  echo "download-r2-object: stripped invisible whitespace/control bytes from R2_ACCESS_KEY_ID ($(value_fingerprint "${raw_access_key_id}"))" >&2
fi
access_key_pattern='^[A-Za-z0-9]+$'
if [[ ! "${R2_ACCESS_KEY_ID}" =~ ${access_key_pattern} ]]; then
  echo "download-r2-object: R2_ACCESS_KEY_ID is not a plausible access key id after sanitization ($(value_fingerprint "${raw_access_key_id}")); refusing to print the value" >&2
  exit 1
fi

raw_secret_access_key="${R2_SECRET_ACCESS_KEY}"
R2_SECRET_ACCESS_KEY="$(strip_invisible_bytes "${raw_secret_access_key}")"
if [[ "${R2_SECRET_ACCESS_KEY}" != "${raw_secret_access_key}" ]]; then
  echo "download-r2-object: stripped invisible whitespace/control bytes from R2_SECRET_ACCESS_KEY (length=$(LC_ALL=C printf '%s' "${raw_secret_access_key}" | wc -c | tr -d '[:space:]'); byte values withheld)" >&2
fi
if [[ -z "${R2_SECRET_ACCESS_KEY}" ]]; then
  echo "download-r2-object: R2_SECRET_ACCESS_KEY is empty after sanitization" >&2
  exit 1
fi

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
