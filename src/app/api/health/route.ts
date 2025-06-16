import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'cival-dashboard-api',
    environment: process.env.NODE_ENV || 'development'
  });
}