import http.server
import socketserver
import os
import argparse

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Redirect / to /mini_scientist/dashboard/index.html
        if self.path == '/':
            self.path = '/mini_scientist/dashboard/index.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dir", type=str, default=".") # Root of the repo
    args = parser.parse_args()

    os.chdir(args.dir)
    pass
    
    Handler = DashboardHandler
    
    print(f"Serving Dashboard at http://localhost:{args.port}")
    print("Accumulation data accessible at /accumulation/global_kg.json")
    
    with socketserver.TCPServer(("", args.port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server")
            httpd.server_close()

if __name__ == "__main__":
    main()
