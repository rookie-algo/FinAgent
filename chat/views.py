# chat/views.py
import asyncio

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions

from agent.agent import run_agent


class SendChatMessageAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]  # adjust to your auth needs

    def post(self, request):
        user = request.user
        message = request.data.get('message')
        result = asyncio.run(run_agent(question=message, user=user))
        # result = await run_agent(question=message, user=user)
        return Response(result, status=status.HTTP_201_CREATED)
