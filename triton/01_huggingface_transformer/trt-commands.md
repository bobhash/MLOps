


```bash
perf_analyzer -m embedder-onnx -u localhost:8500 --concurrency-range 50:50 \
    --shape INPUT_IDS:1,16 \
    --shape ATTENTION_MASK:1,16 \
    --measurement-interval 5000

Concurrency: 1, throughput: 276.178 infer/sec, latency 3619 usec
Concurrency: 2, throughput: 346.176 infer/sec, latency 5775 usec
Concurrency: 3, throughput: 498.364 infer/sec, latency 6017 usec
Concurrency: 4, throughput: 645.18 infer/sec, latency 6197 usec
Concurrency: 5, throughput: 818.19 infer/sec, latency 6109 usec
Concurrency: 6, throughput: 947.216 infer/sec, latency 6332 usec
Concurrency: 7, throughput: 1144.27 infer/sec, latency 6116 usec
Concurrency: 8, throughput: 1301.65 infer/sec, latency 6143 usec
Concurrency: 9, throughput: 1399.52 infer/sec, latency 6428 usec
Concurrency: 10, throughput: 1564.56 infer/sec, latency 6389 usec
Concurrency: 11, throughput: 1702.42 infer/sec, latency 6459 usec
Concurrency: 12, throughput: 1859 infer/sec, latency 6454 usec
Concurrency: 13, throughput: 2004.72 infer/sec, latency 6483 usec
Concurrency: 14, throughput: 2156.48 infer/sec, latency 6490 usec
Concurrency: 15, throughput: 2286.23 infer/sec, latency 6559 usec
Concurrency: 16, throughput: 2411.81 infer/sec, latency 6632 usec
Concurrency: 17, throughput: 2398.34 infer/sec, latency 7087 usec
Concurrency: 18, throughput: 2469.18 infer/sec, latency 7288 usec



perf_analyzer -m embedder-trt -u localhost:8500 --concurrency-range 1:50 \
    --shape INPUT_IDS:1,16 \
    --shape ATTENTION_MASK:1,16 \
    --measurement-interval 5000


```







Usage: perf_analyzer [options]
==== SYNOPSIS ====
 
        --version 
        --service-kind <"triton"|"tfserving"|"torchserve"|"triton_c_api">
        -m <model name>
        -x <model version>
        --bls-composing-models=<string>
        --model-signature-name <model signature name>
        -v

I. MEASUREMENT PARAMETERS: 
        --async (-a)
        --sync
        --measurement-interval (-p) <measurement window (in msec)>
        --concurrency-range <start:end:step>
        --periodic-concurrency-range <start:end:step>
        --request-period <number of responses>
        --request-rate-range <start:end:step>
        --request-distribution <"poisson"|"constant">
        --request-intervals <path to file containing time intervals in microseconds>
        --serial-sequences
        --binary-search
        --num-of-sequences <number of concurrent sequences>
        --latency-threshold (-l) <latency threshold (in msec)>
        --max-threads <thread counts>
        --stability-percentage (-s) <deviation threshold for stable measurement (in percentage)>
        --max-trials (-r)  <maximum number of measurements for each profiling>
        --percentile <percentile>
        DEPRECATED OPTIONS
        -t <number of concurrent requests>
        -c <maximum concurrency>
        -d

II. INPUT DATA OPTIONS: 
        -b <batch size>
        --input-data <"zero"|"random"|<path>>
        --shared-memory <"system"|"cuda"|"none">
        --output-shared-memory-size <size in bytes>
        --shape <name:shape>
        --sequence-length <length>
        --sequence-length-variation <variation>
        --sequence-id-range <start:end>
        --string-length <length>
        --string-data <string>
        --input-tensor-format=[binary|json]
        --output-tensor-format=[binary|json]
        DEPRECATED OPTIONS
        -z
        --data-directory <path>

III. SERVER DETAILS: 
        -u <URL for inference service>
        -i <Protocol used to communicate with inference service>
        --ssl-grpc-use-ssl <bool>
        --ssl-grpc-root-certifications-file <path>
        --ssl-grpc-private-key-file <path>
        --ssl-grpc-certificate-chain-file <path>
        --ssl-https-verify-peer <number>
        --ssl-https-verify-host <number>
        --ssl-https-ca-certificates-file <path>
        --ssl-https-client-certificate-file <path>
        --ssl-https-client-certificate-type <string>
        --ssl-https-private-key-file <path>
        --ssl-https-private-key-type <string>

IV. OTHER OPTIONS: 
        -f <filename for storing report in csv format>
        --profile-export-file <path>
        -H <HTTP header>
        --streaming
        --grpc-compression-algorithm <compression_algorithm>
        --trace-file
        --trace-level
        --trace-rate
        --trace-count
        --log-frequency
        --collect-metrics
        --metrics-url
        --metrics-interval

==== OPTIONS ==== 
 
 --version: print the current version of Perf Analyzer.
 --service-kind: Describes the kind of service perf_analyzer to generate load
         for. The options are "triton", "triton_c_api", "tfserving" and
         "torchserve". Default value is "triton". Note in order to use
         "torchserve" backend --input-data option must point to a json file holding data
         in the following format {"data" : [{"TORCHSERVE_INPUT" :
         ["<complete path to the content file>"]}, {...}...]}. The type of file here
         will depend on the model. In order to use "triton_c_api" you must
         specify the Triton server install path and the model repository path via
         the --triton-server-directory and --model-repository flags
 -m:     This is a required argument and is used to specify the model against
         which to run perf_analyzer.
 -x:     The version of the above model to be used. If not specified the most
         recent version (that is, the highest numbered version) of the model
         will be used.
 --model-signature-name: The signature name of the saved model to use. Default
         value is "serving_default". This option will be ignored if
         --service-kind is not "tfserving".
 -v:     Enables verbose mode.
 -v -v:  Enables extra verbose mode.

I. MEASUREMENT PARAMETERS: 
 --async (-a): Enables asynchronous mode in perf_analyzer. By default,
         perf_analyzer will use synchronous API to request inference. However, if
         the model is sequential then default mode is asynchronous. Specify
         --sync to operate sequential models in synchronous mode. In synchronous
         mode, perf_analyzer will start threads equal to the concurrency
         level. Use asynchronous mode to limit the number of threads, yet
         maintain the concurrency.
 --sync: Force enables synchronous mode in perf_analyzer. Can be used to
         operate perf_analyzer with sequential model in synchronous mode.
 --measurement-interval (-p): Indicates the time interval used for each
         measurement in milliseconds. The perf analyzer will sample a time interval
         specified by -p and take measurement over the requests completed
         within that time interval. The default value is 5000 msec.
 --measurement-mode <"time_windows"|"count_windows">: Indicates the mode used
         for stabilizing measurements. "time_windows" will create windows
         such that the length of each window is equal to --measurement-interval.
         "count_windows" will create windows such that there are at least
         --measurement-request-count requests in each window.
 --measurement-request-count: Indicates the minimum number of requests to be
         collected in each measurement window when "count_windows" mode is
         used. This mode can be enabled using the --measurement-mode flag.
 --concurrency-range <start:end:step>: Determines the range of concurrency
         levels covered by the perf_analyzer. The perf_analyzer will start from
         the concurrency level of 'start' and go till 'end' with a stride of
         'step'. The default value of 'end' and 'step' are 1. If 'end' is not
         specified then perf_analyzer will run for a single concurrency
         level determined by 'start'. If 'end' is set as 0, then the concurrency
         limit will be incremented by 'step' till latency threshold is met.
         'end' and --latency-threshold can not be both 0 simultaneously. 'end'
         can not be 0 for sequence models while using asynchronous mode.
--periodic-concurrency-range <start:end:step>: Determines the range of
         concurrency levels in the similar but slightly different manner as the
         --concurrency-range. Perf Analyzer will start from the concurrency level
         of 'start' and increase by 'step' each time. Unlike
         --concurrency-range, the 'end' indicates the *total* number of concurrency since
         the 'start' (including) and will stop increasing once the cumulative
         number of concurrent requests has reached the 'end'. The user can
         specify *when* to periodically increase the concurrency level using the
         --request-period option. The concurrency level will periodically
         increase for every n-th response specified by --request-period. Since
         this disables stability check in Perf Analyzer and reports response
         timestamps only, the user must provide --profile-export-file to
         specify where to dump all the measured timestamps. The default values of
         'start', 'end', and 'step' are 1.
--request-period <n>: Indicates the number of responses that each request must
         receive before new, concurrent requests are sent when
         --periodic-concurrency-range is specified. Default value is 10.
--request-parameter <name:value:type>: Specifies a custom parameter that can
         be sent to a Triton backend as part of the request. For example,
         providing '--request-parameter max_tokens:256:int' to the command line
         will set an additional parameter 'max_tokens' of type 'int' to 256 as
         part of the request. The --request-parameter may be specified
         multiple times for different custom parameters.
 --request-rate-range <start:end:step>: Determines the range of request rates
         for load generated by analyzer. This option can take floating-point
         values. The search along the request rate range is enabled only when
         using this option. If not specified, then analyzer will search
         along the concurrency-range. The perf_analyzer will start from the
         request rate of 'start' and go till 'end' with a stride of 'step'. The
         default values of 'start', 'end' and 'step' are all 1.0. If 'end' is
         not specified then perf_analyzer will run for a single request rate
         as determined by 'start'. If 'end' is set as 0.0, then the request
         rate will be incremented by 'step' till latency threshold is met.
         'end' and --latency-threshold can not be both 0 simultaneously.
 --request-distribution <"poisson"|"constant">: Specifies the time interval
         distribution between dispatching inference requests to the server.
         Poisson distribution closely mimics the real-world work load on a
         server. This option is ignored if not using --request-rate-range. By
         default, this option is set to be constant.
 --request-intervals: Specifies a path to a file containing time intervals in
         microseconds. Each time interval should be in a new line. The
         analyzer will try to maintain time intervals between successive generated
         requests to be as close as possible in this file. This option can be
         used to apply custom load to server with a certain pattern of
         interest. The analyzer will loop around the file if the duration of
         execution exceeds to that accounted for by the intervals. This option can
         not be used with --request-rate-range or --concurrency-range.
--binary-search: Enables the binary search on the specified search range. This
         option requires 'start' and 'end' to be expilicitly specified in
         the --concurrency-range or --request-rate-range. When using this
         option, 'step' is more like the precision. Lower the 'step', more the
         number of iterations along the search path to find suitable
         convergence. By default, linear search is used.
--num-of-sequences: Sets the number of concurrent sequences for sequence
         models. This option is ignored when --request-rate-range is not
         specified. By default, its value is 4.
 --latency-threshold (-l): Sets the limit on the observed latency. Analyzer
         will terminate the concurrency search once the measured latency
         exceeds this threshold. By default, latency threshold is set 0 and the
         perf_analyzer will run for entire --concurrency-range.
 --max-threads: Sets the maximum number of threads that will be created for
         providing desired concurrency or request rate. However, when runningin
         synchronous mode with concurrency-range having explicit 'end'
         specification,this value will be ignored. Default is 4 if
         --request-rate-range is specified otherwise default is 16.
 --stability-percentage (-s): Indicates the allowed variation in latency
         measurements when determining if a result is stable. The measurement is
         considered as stable if the ratio of max / min from the recent 3
         measurements is within (stability percentage)% in terms of both infer
         per second and latency. Default is 10(%).
 --max-trials (-r): Indicates the maximum number of measurements for each
         concurrency level visited during search. The perf analyzer will take
         multiple measurements and report the measurement until it is stable.
         The perf analyzer will abort if the measurement is still unstable
         after the maximum number of measurements. The default value is 10.
 --percentile: Indicates the confidence value as a percentile that will be
         used to determine if a measurement is stable. For example, a value of
         85 indicates that the 85th percentile latency will be used to
         determine stability. The percentile will also be reported in the results.
         The default is -1 indicating that the average latency is used to
         determine stability
 --serial-sequences: Enables serial sequence mode where a maximum of one
         request is outstanding at a time for any given sequence. The default is
         false.

II. INPUT DATA OPTIONS: 
 -b:     Batch size for each request sent.
 --input-data: Select the type of data that will be used for input in
         inference requests. The available options are "zero", "random", path to a
         directory or a json file. If the option is path to a directory then
         the directory must contain a binary/text file for each
         non-string/string input respectively, named the same as the input. Each file must
         contain the data required for that input for a batch-1 request. Each
         binary file should contain the raw binary representation of the
         input in row-major order for non-string inputs. The text file should
         contain all strings needed by batch-1, each in a new line, listed in
         row-major order. When pointing to a json file, user must adhere to the
         format described in the Performance Analyzer documentation. By
         specifying json data users can control data used with every request.
         Multiple data streams can be specified for a sequence model and the
         analyzer will select a data stream in a round-robin fashion for every
         new sequence. Multiple json files can also be provided (--input-data
         json_file1 --input-data json-file2 and so on) and the analyzer will
         append data streams from each file. When using
         --service-kind=torchserve make sure this option points to a json file. Default is
         "random".
 --shared-memory <"system"|"cuda"|"none">: Specifies the type of the shared
         memory to use for input and output data. Default is none.
 --output-shared-memory-size: The size in bytes of the shared memory region to
         allocate per output tensor. Only needed when one or more of the
         outputs are of string type and/or variable shape. The value should be
         larger than the size of the largest output tensor the model is
         expected to return. The analyzer will use the following formula to
         calculate the total shared memory to allocate: output_shared_memory_size *
         number_of_outputs * batch_size. Defaults to 100KB.
 --shape: The shape used for the specified input. The argument must be
         specified as 'name:shape' where the shape is a comma-separated list for
         dimension sizes, for example '--shape input_name:1,2,3' indicate tensor
         shape [ 1, 2, 3 ]. --shape may be specified multiple times to
         specify shapes for different inputs.
 --sequence-length: Indicates the base length of a sequence used for sequence
         models. A sequence with length X will be composed of X requests to
         be sent as the elements in the sequence. The actual length of the
         sequencewill be within +/- Y% of the base length, where Y defaults to
         20% and is customizable via `--sequence-length-variation`. If
         sequence length is unspecified and input data is provided, the sequence
         length will be the number of inputs in the user-provided input data.
         Default is 20.
 --sequence-length-variation: The percentage variation in length of sequences.
         This flag is only valid when not using user-provided input data or
         when `--sequence-length` is specified while using user-provided
         input data. Default is 20.
 --sequence-id-range <start:end>: Determines the range of sequence id used by
         the perf_analyzer. The perf_analyzer will start from the sequence id
         of 'start' and go till 'end' (excluded). If 'end' is not specified
         then perf_analyzer will use new sequence id without bounds. If 'end'
         is specified and the concurrency setting may result in maintaining
         a number of sequences more than the range of available sequence id,
         perf analyzer will exit with error due to possible sequence id
         collision. The default setting is start from sequence id 1 and without
         bounds
 --string-length: Specifies the length of the random strings to be generated
         by the analyzer for string input. This option is ignored if
         --input-data points to a directory. Default is 128.
 --string-data: If provided, analyzer will use this string to initialize
         string input buffers. The perf analyzer will replicate the given string
         to build tensors of required shape. --string-length will not have any
         effect. This option is ignored if --input-data points to a
         directory.
 --input-tensor-format=[binary|json]: Specifies Triton inference request input
         tensor format. Only valid when HTTP protocol is used. Default is
         'binary'.
 --output-tensor-format=[binary|json]: Specifies Triton inference response
         output tensor format. Only valid when HTTP protocol is used. Default is
         'binary'.

III. SERVER DETAILS: 
 -u:                                  Specify URL to the server. When using triton default is "localhost:8000" if using HTTP and
         "localhost:8001" if using gRPC. When using tfserving default is
         "localhost:8500". 
 -i:                                  The communication protocol to use. The available protocols are gRPC and HTTP. Default is HTTP.
 --ssl-grpc-use-ssl:                  Bool (true|false) for whether to use encrypted channel to the server. Default false.
 --ssl-grpc-root-certifications-file: Path to file containing the PEM encoding of the server root certificates.
 --ssl-grpc-private-key-file:         Path to file containing the PEM encoding of the client's private key.
 --ssl-grpc-certificate-chain-file:   Path to file containing the PEM encoding of the client's certificate chain.
 --ssl-https-verify-peer:             Number (0|1) to verify the peer's SSL certificate. See
         https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYPEER.html for the meaning of each value. Default is 1.
 --ssl-https-verify-host:             Number (0|1|2) to verify the certificate's name against host. See
         https://curl.se/libcurl/c/CURLOPT_SSL_VERIFYHOST.html for the meaning of each value. Default is 2.
 --ssl-https-ca-certificates-file:    Path to Certificate Authority (CA) bundle.
 --ssl-https-client-certificate-file: Path to the SSL client certificate.
 --ssl-https-client-certificate-type: Type (PEM|DER) of the client SSL certificate. Default is PEM.
 --ssl-https-private-key-file:        Path to the private keyfile for TLS and SSL client cert.
 --ssl-https-private-key-type:        Type (PEM|DER) of the private key file. Default is PEM.

IV. OTHER OPTIONS: 
 -f:     The latency report will be stored in the file named by this option.
         By default, the result is not recorded in a file.
 --profile-export-file: Specifies the path that the profile export will be generated at. By
         default, the profile export will not be generated.
 -H:     The header will be added to HTTP requests (ignored for GRPC
         requests). The header must be specified as 'Header:Value'. -H may be
         specified multiple times to add multiple headers.
 --streaming: Enables the use of streaming API. This flag is only valid with
         gRPC protocol. By default, it is set false.
 --grpc-compression-algorithm: The compression algorithm to be used by gRPC
         when sending request. Only supported when grpc protocol is being used.
         The supported values are none, gzip, and deflate. Default value is
         none.
 --trace-file: Set the file where trace output will be saved. If
         --log-frequency is also specified, this argument value will be the prefix of the
         files to save the trace output. See --log-frequency for details.
         Only used for service-kind of triton. Default value is none.
 --trace-level: Specify a trace level. OFF to disable tracing, TIMESTAMPS to
         trace timestamps, TENSORS to trace tensors. It may be specified
         multiple times to trace multiple information. Default is OFF.
 --trace-rate: Set the trace sampling rate. Default is 1000.
 --trace-count: Set the number of traces to be sampled. If the value is -1,
         the number of traces to be sampled will not be limited. Default is -1.
 --log-frequency:  Set the trace log frequency. If the value is 0, Triton will
         only log the trace output to <trace-file> when shutting down.
         Otherwise, Triton will log the trace output to <trace-file>.<idx> when it
         collects the specified number of traces. For example, if the log
         frequency is 100, when Triton collects the 100-th trace, it logs the
         traces to file <trace-file>.0, and when it collects the 200-th trace,
         it logs the 101-th to the 200-th traces to file <trace-file>.1.
         Default is 0.
 --triton-server-directory: The Triton server install path. Required by and
         only used when C API is used (--service-kind=triton_c_api).
         eg:--triton-server-directory=/opt/tritonserver.
 --model-repository: The model repository of which the model is loaded.
         Required by and only used when C API is used
         (--service-kind=triton_c_api). eg:--model-repository=/tmp/host/docker-data/model_unit_test.
 --verbose-csv: The csv files generated by perf analyzer will include
         additional information.
 --collect-metrics: Enables collection of server-side inference server
         metrics. Outputs metrics in the csv file generated with the -f option. Must
         enable `--verbose-csv` option to use the `--collect-metrics`.
 --metrics-url: The URL to query for server-side inference server metrics.
         Default is 'localhost:8002/metrics'.
 --metrics-interval: How often in milliseconds, within each measurement
         window, to query for server-side inference server metrics. Default is
         1000.
 --bls-composing-models: A comma separated list of all BLS composing models
         (with optional model version number after a colon for each) that may
         be called by the input BLS model. For example, 'modelA:3,modelB'
         would specify that modelA and modelB are composing models that may be
         called by the input BLS model, and that modelA will use version 3,
         while modelB's version is unspecified