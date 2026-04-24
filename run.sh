#!/bin/bash

# Function to safely change directory
cd_safe() {
    if ! cd "$1"; then
        echo "Error: Failed to change directory to $1"
        exit 1
    fi
}

ensure_network() {
    if ! docker network inspect aio-network &>/dev/null; then
        echo "Creating aio-network..."
        docker network create aio-network
    fi
}

up_backend() {
    echo "Starting backend service..."
    ensure_network
    cd_safe backend
    docker compose up -d
    echo "Backend service is up."
    cd_safe ..
}

up_frontend() {
    echo "Starting frontend service..."
    ensure_network
    cd_safe frontend
    docker compose up -d
    echo "Frontend service is up."
    cd_safe ..
}

up_vllm() {
    echo "Starting VLLM service..."
    ensure_network
    cd_safe vllm_api
    docker compose up -d
    echo "VLLM service is up."
    cd_safe ..
}

up_trtllm() {
    echo "Starting TensorRT-LLM service..."
    ensure_network
    cd_safe trtllm_api
    docker compose up -d
    echo "TensorRT-LLM service is up."
    cd_safe ..
}

up_monitor() {
    echo "Starting monitoring services..."
    ensure_network
    cd_safe monitor
    docker compose up -d
    echo "Monitoring services are up."
    cd_safe ..
}

down_backend() {
    echo "Stopping backend service..."
    cd_safe backend
    docker compose down
    echo "Backend service is stopped."
    cd_safe ..
}

down_frontend() {
    echo "Stopping frontend service..."
    cd_safe frontend
    docker compose down
    echo "Frontend service is stopped."
    cd_safe ..
}

down_vllm() {
    echo "Stopping VLLM service..."
    cd_safe vllm_api
    docker compose down
    echo "VLLM service is stopped."
    cd_safe ..
}

down_trtllm() {
    echo "Stopping TensorRT-LLM service..."
    cd_safe trtllm_api
    docker compose down
    echo "TensorRT-LLM service is stopped."
    cd_safe ..
}

down_monitor() {
    echo "Stopping monitoring services..."
    cd_safe monitor
    docker compose down
    echo "Monitoring services are stopped."
    cd_safe ..
}

up_services() {
    # Start services in proper order
    up_monitor
    up_vllm
    up_trtllm
    up_backend
    up_frontend
    
    echo "All services are up and running."
}

up_app_stack() {
    up_monitor
    up_backend
    up_frontend

    echo "Application stack is up."
}

down_app_stack() {
    down_frontend
    down_backend
    down_monitor

    echo "Application stack is stopped."
}

down_services() {
    # Stop services in reverse order
    down_frontend
    down_backend
    down_trtllm
    down_vllm
    down_monitor
    
    echo "All services have been stopped."
}

restart_services() {
    echo "Restarting all services..."
    down_services
    up_services
}

status_services() {
    echo "Checking status of all services..."
    
    echo "=== Backend Services ==="
    cd_safe backend
    docker compose ps
    cd_safe ..
    
    echo "=== Frontend Services ==="
    cd_safe frontend
    docker compose ps
    cd_safe ..
    
    echo "=== VLLM Services ==="
    cd_safe vllm_api
    docker compose ps
    cd_safe ..

    echo "=== TensorRT-LLM Services ==="
    cd_safe trtllm_api
    docker compose ps
    cd_safe ..
    
    echo "=== Monitoring Services ==="
    cd_safe monitor
    docker compose ps
    cd_safe ..
}

help() {
    echo "Usage: $0 {up|down|restart|status|help}"
    echo "Commands:"
    echo "  up          Start all services on one machine"
    echo "  down        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show status of all services"
    echo "  up-app      Start monitor, backend, and frontend"
    echo "  down-app    Stop monitor, backend, and frontend"
    echo "  up-vllm     Start only the vLLM node service"
    echo "  down-vllm   Stop only the vLLM node service"
    echo "  up-trtllm   Start only the TensorRT-LLM node service"
    echo "  down-trtllm Stop only the TensorRT-LLM node service"
    echo "  help        Show this help message"
}

# Main script logic
if [ $# -eq 0 ]; then
    help
    exit 1
fi

case "$1" in
    up)
        up_services
        ;;
    down)
        down_services
        ;;
    restart)
        restart_services
        ;;
    status)
        status_services
        ;;
    up-app)
        up_app_stack
        ;;
    down-app)
        down_app_stack
        ;;
    up-vllm)
        up_vllm
        ;;
    down-vllm)
        down_vllm
        ;;
    up-trtllm)
        up_trtllm
        ;;
    down-trtllm)
        down_trtllm
        ;;
    help)
        help
        ;;
    *)
        echo "Invalid command: $1"
        help
        exit 1
        ;;
esac