#!/bin/sh

# Name:             get_WL_data.sh
# Purpose:          Downloads 2021-2026 NOAA CO-OPS Water Level Data in Monthly Chunks
# Output Directory: ../storm_surge/

BEGINDATE_YEAR=2021
ENDDATE_YEAR=2026
DATUM="mllw"
UNIT="english"
FORM="csv"

# Target output directory
OUTPUT_DIR="../storm_surge"
CONFIG_FILE="stations_config.csv"

if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="$OUTPUT_DIR/stations_config.csv"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Master station configuration file missing at $CONFIG_FILE"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "📖 Parsing station registry: $CONFIG_FILE"
STATION_IDS=$(awk -F',' 'NR>1 {print $1}' "$CONFIG_FILE" | tr -d ' "\r')

for STATIONID in $STATION_IDS; do
    [ -z "$STATIONID" ] && continue

    TARGET_FILE="$OUTPUT_DIR/wl_${STATIONID}.csv"
    TEMP_CHUNK="$OUTPUT_DIR/temp_${STATIONID}.csv"
    
    echo "📍 Processing Station $STATIONID -> $TARGET_FILE"
    
    # Remove old target file to start clean
    rm -f "$TARGET_FILE" "$TEMP_CHUNK"

    # Loop year by year and month by month to satisfy NOAA 31-day limit
    for YR in $(seq $BEGINDATE_YEAR $ENDDATE_YEAR); do
        for M in $(seq -w 1 12); do
            # Compute last day of the month
            case $M in
                01|03|05|07|08|10|12) LAST_DAY="31" ;;
                04|06|09|11)          LAST_DAY="30" ;;
                02) 
                    if [ $((YR % 4)) -eq 0 ] && { [ $((YR % 100)) -ne 0 ] || [ $((YR % 400)) -eq 0 ]; }; then
                        LAST_DAY="29"
                    else
                        LAST_DAY="28"
                    fi
                    ;;
            esac

            B_DATE="${YR}${M}01"
            E_DATE="${YR}${M}${LAST_DAY}"

            # Query product=hourly_height for direct hourly water levels
            REQUESTGET="https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?product=hourly_height&application=NOS.COOPS.TAC.WL&begin_date=${B_DATE}&end_date=${E_DATE}&station=${STATIONID}&datum=${DATUM}&units=${UNIT}&time_zone=gmt&application=MDL&format=${FORM}"

            curl -s "$REQUESTGET" -o "$TEMP_CHUNK"

            # Check if chunk contains valid data rows (ignoring header and API error text)
            if grep -q "Date Time" "$TEMP_CHUNK" || grep -q "202" "$TEMP_CHUNK"; then
                if [ ! -f "$TARGET_FILE" ]; then
                    # Write header + data for first successful chunk
                    cat "$TEMP_CHUNK" > "$TARGET_FILE"
                else
                    # Append data rows only (skip header) for subsequent chunks
                    tail -n +2 "$TEMP_CHUNK" >> "$TARGET_FILE"
                fi
            fi

            rm -f "$TEMP_CHUNK"
            sleep 0.05
        done
    done

    if [ -f "$TARGET_FILE" ]; then
        ROW_COUNT=$(wc -l < "$TARGET_FILE")
        echo "  ✅ Station $STATIONID complete: Saved $ROW_COUNT records to $TARGET_FILE"
    else
        echo "  ⚠️ Station $STATIONID yielded zero valid observation records."
    fi
done

echo "🎉 All station downloads completed successfully."
