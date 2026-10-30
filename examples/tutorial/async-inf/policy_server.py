from lerobot.async_inference.server.policy_server import serve
from lerobot.async_inference.shared.configs import PolicyServerConfig


def main():
    host = ...  # something like "127.0.0.1" if you're exposing to localhost
    port = ...  # something like 8080

    config = PolicyServerConfig(
        host=host,
        port=port,
    )
    serve(config)


if __name__ == "__main__":
    main()
